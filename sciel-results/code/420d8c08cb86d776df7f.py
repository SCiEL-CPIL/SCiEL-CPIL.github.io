import numpy as np
import time
import sys

def _build_hamiltonians():
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def two_site_op(op1, site1, op2, site2):
        ops = []
        for i in range(N):
            if i == site1:
                ops.append(op1)
            elif i == site2:
                ops.append(op2)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond_term(site1, site2):
        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def _nelder_mead(f, x0, max_iter=1000, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-8, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    n = len(x0)
    
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.05 if x_new[i] == 0 else 0.05 * x_new[i]
        simplex.append(x_new)
        
    values = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[0] - values[-1]) < tol:
            break
            
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < values[0]:
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                for i in range(1, n+1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                    
    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0
    
    A, B = _build_hamiltonians()
    H = A + B
    
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals_H)
        U_t = eigvecs_H @ np.diag(phase) @ eigvecs_H.conj().T
        U_exact.append(U_t)
        
    def get_exp_A(t_coeff):
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_A)
        return eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
        
    def get_exp_B(t_coeff):
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_B)
        return eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T

    def objective(params):
        a_free = params[:11]
        b_free = params[11:22]
        
        a = np.array(a_free)
        a_full = np.concatenate([a, [1.0 - np.sum(a)]])
        
        b = np.array(b_free)
        b_full = np.concatenate([b, [1.0 - np.sum(b)]])
        
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
            
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                E_A_k = get_exp_A(a_full[k] * t)
                E_B_k = get_exp_B(b_full[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
                
            diff = U_approx - U_exact[i]
            err = np.linalg.norm(diff, 2)
            err_sum += err
            
        return err_sum / len(times)
        
    # Suzuki-Trotter 4th order coefficients
    x = 2**(1/3)
    a1 = 1 / (2 * (1 - x))
    a2 = 1 / (2 - x)
    b1 = 1 / (1 - x)
    b2 = -x / (1 - x)
    
    # Pattern: B(b1) A(a1) B(b2) A(a2) B(b1) A(a1)
    # This is 3 A's and 3 B's.
    # We need 12 A's and 12 B's.
    # Repeat 4 times.
    a_pattern = [a1, a2, a1]
    b_pattern = [b1, b2, b1]
    
    a_init = []
    b_init = []
    for _ in range(4):
        a_init.extend(a_pattern)
        b_init.extend(b_pattern)
        
    # We have 12 a's and 12 b's.
    # We need to map to 11 free a's and 11 free b's.
    a_free_init = a_init[:11]
    b_free_init = b_init[:11]
    
    x0 = np.array(a_free_init + b_free_init)
    
    x_best, val_best = _nelder_mead(objective, x0, max_iter=3000, time_limit=130.0, start_time=start_time)
    
    a_free = x_best[:11]
    b_free = x_best[11:22]
    
    a = list(a_free)
    a.append(1.0 - np.sum(a_free))
    
    b = list(b_free)
    b.append(1.0 - np.sum(b_free))
    
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    return a, b

if __name__ == "__main__":
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
