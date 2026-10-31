import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin from 210s limit

    # Indices for Hermite polynomials
    indices = [0, 4, 8, 12]
    n = len(indices)
    
    # Precompute M_ii = sqrt(pi) * 2^i * i!
    # and Q_ii, Q_ij terms
    # M is diagonal
    M = np.zeros((n, n))
    Q = np.zeros((n, n))
    
    for idx, i in enumerate(indices):
        # M_ii
        M[idx, idx] = np.sqrt(np.pi) * (2 ** i) * np.math.factorial(i)
        
        # Q_ii = (i + 0.5) * M_ii
        Q[idx, idx] = (i + 0.5) * M[idx, idx]
        
        # Off-diagonal terms for Q
        # Q_ij is non-zero if |i-j| == 2
        # Formula: Q_ij = 1/4 M_{i, j+2} if i = j+2
        #           Q_ij = j(j-1)/2 M_{i, j-2} if i = j-2
        # Note: M_ab is non-zero only if a=b.
        
        for jdx, j in enumerate(indices):
            if i == j + 2:
                # i = j+2 => M_{i, i} is used
                Q[idx, jdx] = 0.25 * M[idx, idx]
            elif i == j - 2:
                # i = j-2 => M_{i, i} is used
                Q[idx, jdx] = 0.5 * j * (j - 1) * M[idx, idx]
            # Symmetry Q_ij = Q_ji
            # The derivation x^2 H_j = ... H_i ... 
            # Let's check symmetry.
            # \int x^2 H_i H_j = \int H_i (x^2 H_j).
            # Also \int H_j (x^2 H_i).
            # The operator x^2 is symmetric. So Q must be symmetric.
            # My formula:
            # If i = j+2, Q_ij = 0.25 M_ii.
            # If j = i+2 (so i = j-2), Q_ji = 0.5 * i * (i-1) M_jj? No.
            # Let's re-derive carefully.
            # \int x^2 H_i H_j dx.
            # Use x^2 H_j = 1/4 H_{j+2} + (j+0.5) H_j + j(j-1)/2 H_{j-2}.
            # Integral = 1/4 \int H_i H_{j+2} + (j+0.5) \int H_i H_j + j(j-1)/2 \int H_i H_{j-2}.
            # Since \int H_a H_b = M_ab \delta_{ab}:
            # Term 1: 1/4 M_{i, j+2} if i = j+2. Value: 1/4 M_{i,i}.
            # Term 2: (j+0.5) M_{ij} if i=j. Value: (j+0.5) M_{jj}.
            # Term 3: j(j-1)/2 M_{i, j-2} if i = j-2. Value: j(j-1)/2 M_{i,i}.
            
            # So if i = j+2: Q_ij = 0.25 M_{i,i}.
            # If i = j-2: Q_ij = j(j-1)/2 M_{i,i}.
            
            # Check symmetry:
            # Case i = j+2. Q_ij = 0.25 M_{i,i}.
            # What is Q_ji? Here j = i-2.
            # Using formula for Q_ji with "i" replaced by j and "j" replaced by i:
            # Q_ji (where first index is j, second is i, and j = i-2)
            # Here "current_i" = j, "current_j" = i.
            # Condition "current_i = current_j - 2" is j = i - 2. True.
            # So Q_ji = i(i-1)/2 M_{j,j}.
            
            # Are they equal?
            # Q_ij = 0.25 M_{i,i} = 0.25 \sqrt{\pi} 2^i i!
            # Q_ji = i(i-1)/2 M_{j,j} = i(i-1)/2 \sqrt{\pi} 2^j j!
            # Since i = j+2:
            # M_{i,i} = \sqrt{\pi} 2^{j+2} (j+2)! = 4 \cdot 2^j \cdot (j+2)(j+1) j!
            # Q_ij = 0.25 \cdot 4 \cdot 2^j (j+2)(j+1) j! \sqrt{\pi} = (j+2)(j+1) 2^j j! \sqrt{\pi}
            
            # M_{j,j} = \sqrt{\pi} 2^j j!
            # Q_ji = (j+2)(j+1)/2 \cdot \sqrt{\pi} 2^j j! = 0.5 (j+2)(j+1) 2^j j! \sqrt{\pi}
            
            # They differ by a factor of 2!
            # Q_ij = 2 * Q_ji.
            # This implies my formula or assumption is wrong.
            # Let's check \int x^2 H_2 H_0.
            # H_2 = 4x^2 - 2. H_0 = 1.
            # \int x^2 (4x^2-2) e^{-x^2} = 4 \int x^4 e^{-x^2} - 2 \int x^2 e^{-x^2}.
            # \int x^2 e^{-x^2} = \sqrt{\pi}/2.
            # \int x^4 e^{-x^2} = 3\sqrt{\pi}/4.
            # Val = 4(3\sqrt{\pi}/4) - 2(\sqrt{\pi}/2) = 3\sqrt{\pi} - \sqrt{\pi} = 2\sqrt{\pi}.
            
            # Using formula for i=2, j=0 (i=j+2):
            # Q_20 = 0.25 M_{2,2} = 0.25 \sqrt{\pi} 2^2 2! = 0.25 \sqrt{\pi} 8 = 2\sqrt{\pi}.
            # Matches.
            
            # Using formula for i=0, j=2 (i=j-2):
            # Q_02 = j(j-1)/2 M_{0,0} = 2(1)/2 \sqrt{\pi} 1 = \sqrt{\pi}.
            # Mismatch! Q_20 = 2\sqrt{\pi}, Q_02 = \sqrt{\pi}.
            # But the integral \int x^2 H_0 H_2 should be the same as \int x^2 H_2 H_0.
            # So Q must be symmetric.
            # Where is the error?
            # Recurrence: x^2 H_n = 1/4 H_{n+2} + (n+0.5) H_n + n(n-1)/2 H_{n-2}.
            # For n=0: x^2 H_0 = 1/4 H_2 + 0.5 H_0.
            # \int x^2 H_0 H_2 = \int (1/4 H_2 + 0.5 H_0) H_2 = 1/4 M_{2,2} + 0.
            # = 1/4 (2\sqrt{\pi} \cdot 2) = \sqrt{\pi}.
            # Wait, M_{2,2} = 2\sqrt{\pi} \cdot 2 = 4\sqrt{\pi}?
            # M_{nn} = \sqrt{\pi} 2^n n!.
            # M_{0,0} = \sqrt{\pi}.
            # M_{2,2} = \sqrt{\pi} 4 (2) = 8\sqrt{\pi}.
            # So Q_02 = 1/4 (8\sqrt{\pi}) = 2\sqrt{\pi}.
            # Matches Q_20.
            
            # My previous calculation of M_{2,2} was wrong in the head.
            # M_{2,2} = 8\sqrt{\pi}.
            # Q_20 = 0.25 * 8\sqrt{\pi} = 2\sqrt{\pi}.
            # Q_02 = 2(1)/2 * M_{0,0} = 1 * \sqrt{\pi} = \sqrt{\pi}.
            # Still mismatch.
            
            # Let's re-evaluate \int x^2 H_2 H_0.
            # H_2 = 4x^2 - 2.
            # x^2 H_2 = 4x^4 - 2x^2.
            # \int (4x^4 - 2x^2) e^{-x^2} = 4(3\sqrt{\pi}/4) - 2(\sqrt{\pi}/2) = 3\sqrt{\pi} - \sqrt{\pi} = 2\sqrt{\pi}.
            
            # Now \int x^2 H_0 H_2.
            # H_0 = 1.
            # x^2 H_0 = x^2.
            # \int x^2 H_2 e^{-x^2} = \int x^2 (4x^2-2) e^{-x^2} = 2\sqrt{\pi}.
            
            # So both are 2\sqrt{\pi}.
            # Why did the formula give \sqrt{\pi} for Q_02?
            # Formula for i=0, j=2 (i=j-2):
            # Term 3: j(j-1)/2 M_{i, j-2} = 2(1)/2 M_{0, 0} = 1 \cdot \sqrt{\pi} = \sqrt{\pi}.
            # Is there another term?
            # x^2 H_j with j=2.
            # x^2 H_2 = 1/4 H_4 + 2.5 H_2 + 1 H_0.
            # \int H_0 (1/4 H_4 + 2.5 H_2 + 1 H_0) = \int (1/4 H_4 H_0 + 2.5 H_2 H_0 + 1 H_0^2).
            # Only H_0^2 term survives.
            # Val = 1 * M_{0,0} = \sqrt{\pi}.
            
            # Contradiction!
            # \int x^2 H_2 H_0 = 2\sqrt{\pi}.
            # \int H_0 x^2 H_2 = \sqrt{\pi}.
            # This implies x^2 is not a symmetric operator in this basis?
            # No, x^2 is symmetric.
            # Ah, the recurrence $x^2 H_n$ is correct.
            # But $\int H_i H_j$ is the inner product.
            # $\langle H_i, x^2 H_j \rangle$.
            # Since $x^2$ is symmetric, $\langle H_i, x^2 H_j \rangle = \langle x^2 H_i, H_j \rangle$.
            # So Q_ij should be symmetric.
            # My calculation of $\int x^2 H_2 H_0$ gave $2\sqrt{\pi}$.
            # My calculation of $\int H_0 x^2 H_2$ gave $\sqrt{\pi}$.
            # Let's re-calculate $\int x^2 H_2 H_0$ directly.
            # $H_2 = 4x^2-2$.
            # $x^2 H_2 = 4x^4 - 2x^2$.
            # $\int (4x^4 - 2x^2) e^{-x^2} dx$.
            # $\int x^4 e^{-x^2} = 3\sqrt{\pi}/4$.
            # $\int x^2 e^{-x^2} = \sqrt{\pi}/2$.
            # $4(3\sqrt{\pi}/4) - 2(\sqrt{\pi}/2) = 3\sqrt{\pi} - \sqrt{\pi} = 2\sqrt{\pi}$.
            
            # Now $\int H_0 x^2 H_2$.
            # $x^2 H_2 = 4x^4 - 2x^2$.
            # $\int 1 \cdot (4x^4 - 2x^2) e^{-x^2} dx = 2\sqrt{\pi}$.
            
            # So both are $2\sqrt{\pi}$.
            # Where did the recurrence formula fail?
            # $x^2 H_2 = 1/4 H_4 + 2.5 H_2 + 1 H_0$.
            # $\int H_0 (1/4 H_4 + 2.5 H_2 + 1 H_0) e^{-x^2}$.
            # $= 1/4 \int H_0 H_4 + 2.5 \int H_0 H_2 + 1 \int H_0 H_0$.
            # $= 0 + 0 + 1 \cdot \sqrt{\pi} = \sqrt{\pi}$.
            
            # The recurrence must be wrong.
            # $x H_n = 1/2 H_{n+1} + n H_{n-1}$.
            # $x^2 H_n = x (1/2 H_{n+1} + n H_{n-1})$.
            # $= 1/2 (1/2 H_{n+2} + (n+1) H_n) + n (1/2 H_n + (n-1) H_{n-2})$.
            # $= 1/4 H_{n+2} + (n+1)/2 H_n + n/2 H_n + n(n-1)/2 H_{n-2}$.
            # $= 1/4 H_{n+2} + (n + 0.5) H_n + n(n-1)/2 H_{n-2}$.
            
            # For n=2:
            # $x^2 H_2 = 1/4 H_4 + 2.5 H_2 + 1 H_0$.
            # $H_4 = 16x^4 - 48x^2 + 12$.
            # $1/4 H_4 = 4x^4 - 12x^2 + 3$.
            # $2.5 H_2 = 2.5 (4x^2 - 2) = 10x^2 - 5$.
            # $1 H_0 = 1$.
            # Sum: $4x^4 - 12x^2 + 3 + 10x^2 - 5 + 1 = 4x^4 - 2x^2 - 1$.
            # But $x^2 H_2 = 4x^4 - 2x^2$.
            # There is a $-1$ term.
            # The recurrence $x H_n = 1/2 H_{n+1} + n H_{n-1}$ is for **probabilists'** Hermite polynomials?
            # Physicists': $H_{n+1} = 2x H_n - 2n H_{n-1}$.
            # So $2x H_n = H_{n+1} + 2n H_{n-1}$.
            # $x H_n = 1/2 H_{n+1} + n H_{n-1}$.
            # This is correct.
            
            # Why the discrepancy?
            # $H_0 = 1$.
            # $H_1 = 2x$.
            # $H_2 = 4x^2 - 2$.
            # $H_3 = 8x^3 - 12x$.
            # $H_4 = 16x^4 - 48x^2 + 12$.
            
            # Check $x H_1 = 2x^2$.
            # Formula: $1/2 H_2 + 1 H_0 = 1/2 (4x^2-2) + 1 = 2x^2 - 1 + 1 = 2x^2$. Correct.
            
            # Check $x H_2 = 4x^3 - 2x$.
            # Formula: $1/2 H_3 + 2 H_1 = 1/2 (8x^3-12x) + 2(2x) = 4x^3 - 6x + 4x = 4x^3 - 2x$. Correct.
            
            # So $x^2 H_2$ calculation:
            # $x (4x^3 - 2x) = 4x^4 - 2x^2$.
            # Formula result: $4x^4 - 2x^2 - 1$.
            # The formula gave an extra $-1$.
            # $1/4 H_4 + 2.5 H_2 + 1 H_0$.
            # $1/4 (16x^4 - 48x^2 + 12) + 2.5 (4x^2 - 2) + 1$.
            # $4x^4 - 12x^2 + 3 + 10x^2 - 5 + 1 = 4x^4 - 2x^2 - 1$.
            # The sum of constants is $3 - 5 + 1 = -1$.
            # It should be 0.
            
            # The recurrence $x^2 H_n$ derived from $x H_n$ is correct.
            # The issue is that $H_n$ are not orthonormal?
            # No, they are orthogonal.
            # The issue is that the expansion of $x^2 H_n$ in terms of $H_k$ is unique.
            # So $x^2 H_2 = 1/4 H_4 + 2.5 H_2 + 1 H_0$ is WRONG.
            # It must be $x^2 H_2 = 1/4 H_4 + 2.5 H_2 + 1 H_0 - 1$?
            # No, $H_0=1$. So $1 H_0$ is 1.
            # The constant term in $x^2 H_2$ is 0.
            # The constant term in the RHS is $-1$.
            # So the coefficient of $H_0$ must be 0?
            # If coeff of $H_0$ is 0, constant term is $3-5 = -2$. No.
            
            # Let's just use numerical quadrature for the matrices M and Q to be safe.
            # Gauss-Hermite quadrature is exact for polynomials.
            pass

    # Use Gauss-Hermite quadrature
    # Nodes and weights for e^{-x^2}
    # np.polynomial.hermite.hermgauss
    x, w = np.polynomial.hermite.hermgauss(100)
    # w_i * f(x_i) approximates \int f(x) e^{-x^2} dx
    
    def H(n, x):
        return np.polynomial.hermite.hermval(x, np.polynomial.hermite.herm2phys(np.polynomial.hermite.herm2phys(np.zeros(n+1)))) # Too complex
        # Better: use explicit coefficients or recurrence
        # np.polynomial.hermite.hermval(x, coeffs)
        # coeffs for H_n:
        # H_0 = [1]
        # H_1 = [0, 2]
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        
    # Precompute H_i(x) for i in indices
    H_vals = {}
    for i in indices:
        # Generate coefficients for H_i
        c = np.zeros(i+1)
        if i == 0: c[0] = 1
        elif i == 1: c[1] = 2
        else:
            # Recurrence
            c_prev = np.zeros(i)
            c_prev2 = np.zeros(i-1)
            if i >= 1:
                c_prev[1] = 2 # H_1
            if i >= 2:
                # H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2
                c[2] = 4
                c[0] = -2
            # This is getting messy. Use scipy
            pass
        
    # Simpler: use scipy.special.eval_hermite
    from scipy.special import eval_hermite
    
    H_vals = {i: eval_hermite(i, x) for i in indices}
    
    for idx, i in enumerate(indices):
        for jdx, j in enumerate(indices):
            # M_ij
            M[idx, jdx] = np.sum(w * H_vals[i] * H_vals[j])
            # Q_ij
            Q[idx, jdx] = np.sum(w * x**2 * H_vals[i] * H_vals[j])
            
    # Now optimize
    best_val = np.inf
    best_c = None
    
    def get_c_full(c_free):
        c0, c4, c8 = c_free
        c12 = (c0 - 3*c4 + 105*c8) / 10395.0
        return np.array([c0, c4, c8, c12])
        
    def objective(c_free):
        c = get_c_full(c_free)
        I0 = c @ M @ c
        I2 = c @ Q @ c
        if I0 <= 0: return 1e10
        val = (I2 / I0) ** 2
        return val
        
    def gradient(c_free):
        c = get_c_full(c_free)
        I0 = c @ M @ c
        I2 = c @ Q @ c
        if I0 <= 0: return np.zeros(3)
        
        # dV/dc = 2 (I2/I0) / I0 * (I0 dI2/dc - I2 dI0/dc)
        # dI0/dc = 2 M c
        # dI2/dc = 2 Q c
        dI0_dc = 2 * (M @ c)
        dI2_dc = 2 * (Q @ c)
        
        grad_c = 2 * (I2 / I0) / I0 * (I0 * dI2_dc - I2 * dI0_dc)
        # grad_c is w.r.t c_full
        # Map to c_free
        # c_full = [c0, c4, c8, c12]
        # c12 = (c0 - 3c4 + 105c8) / 10395
        # dc12/dc0 = 1/10395
        # dc12/dc4 = -3/10395
        # dc12/dc8 = 105/10395
        
        J = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1/10395, -3/10395, 105/10395]
        ])
        
        grad_free = J.T @ grad_c
        return grad_free

    # Restart loop
    np.random.seed(42)
    for i in range(100):
        if time.time() > deadline: break
        c0 = np.random.normal(0, 1)
        c4 = np.random.normal(0, 0.1)
        c8 = np.random.normal(0, 0.01)
        c_init = np.array([c0, c4, c8])
        
        res = minimize(objective, c_init, jac=gradient, method='L-BFGS-B', options={'maxiter': 1000})
        
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x

    return best_c

# EVOLVE-BLOCK-END
