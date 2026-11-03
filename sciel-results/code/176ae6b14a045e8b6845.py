import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s hard limit
    
    # Helper to calculate cost
    def calc_cost(p):
        # p is a permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # This is trace(a @ b[p][:,p]) or sum(a * b[np.ix_(p,p)])
        return np.sum(a * b[np.ix_(p, p)])

    # Helper to calculate delta of swapping positions i and j in permutation p
    def delta_2opt(p, i, j):
        u = p[i]
        v = p[j]
        
        # Precompute differences
        a_i_j = a[i, :] - a[j, :]
        a_k_i_j = a[:, i] - a[:, j]
        
        b_v_u = b[v, :] - b[u, :]
        b_v_u_col = b[:, v] - b[:, u]
        
        # Full sums
        s1 = np.dot(a_i_j, b_v_u)
        s2 = np.dot(a_k_i_j, b_v_u_col)
        
        # Corrections for k=i and k=j in s1
        # k=i: (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
        c1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # k=j: (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
        c1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        # Corrections for k=i and k=j in s2
        # k=i: (a[i,i]-a[i,j]) * (b[u,v] - b[u,u])
        c2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        # k=j: (a[j,i]-a[j,j]) * (b[v,v] - b[v,u])
        c2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        # Explicit terms for i,j pair
        # a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v])
        # a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        explicit = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + \
                   (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        delta = s1 + s2 - c1_i - c1_j - c2_i - c2_j + explicit
        return delta

    # Phase 1: Initial Solution
    # Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Quick 2-opt polish on initial solution
    polish_deadline = start_time + 10.0
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for i in range(n):
            if time.time() > polish_deadline:
                break
            best_j = -1
            best_d = 0
            for j in range(i+1, n):
                d = delta_2opt(p, i, j)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
    
    # Phase 2: Frank-Wolfe
    fw_deadline = start_time + 40.0
    # Initialize X as permutation matrix
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p] = 1.0
    
    # We will run FW for a while, updating best_p if we find a better permutation
    # Gradient of f(X) = tr(A X B X^T) is G = A X B + A^T X B^T
    # Since we are dealing with real matrices, and X is a permutation matrix (which is a vertex),
    # the FW step is:
    # 1. Compute Gradient G
    # 2. Solve LAP on G to get Y (a permutation matrix)
    # 3. Update X = (1-alpha) X + alpha Y
    # 4. Extract permutation from Y (or the closest vertex)
    
    # Note: The FW method works on the convex hull. The "permutation" we extract is from the vertex Y.
    # We can also try to extract a permutation from X by rounding or solving a LAP on X itself? 
    # Usually, we just take the permutation corresponding to Y.
    
    t = 0
    while time.time() < fw_deadline:
        t += 1
        # Compute Gradient
        # G = A X B + A.T X B.T
        G = a @ X @ b + a.T @ X @ b.T
        G = (G + G.T) / 2.0  # Symmetrize to ensure real LAP solution? Not strictly necessary but good practice
        
        # Solve Linear Assignment Problem
        # We want to minimize sum G[i,j] P[i,j]
        row_ind, col_ind = linear_sum_assignment(G)
        
        # Create Y
        Y = np.zeros((n, n), dtype=np.float64)
        Y[row_ind, col_ind] = 1.0
        
        # Get permutation from Y
        p_fw = np.empty(n, dtype=np.int64)
        p_fw[row_ind] = col_ind
        
        cost_fw = calc_cost(p_fw)
        if cost_fw < best_cost:
            best_cost = cost_fw
            best_p = p_fw.copy()
            # Update current p to this new best for SA later? 
            # Or just keep best_p. Let's update p to track the "current" FW state's best vertex.
            p = p_fw.copy()
            current_cost = cost_fw
            
        # Update X
        alpha = 2.0 / (t + 2)
        X = (1 - alpha) * X + alpha * Y
        
        # Optional: Check if X is close to a vertex?
        # If so, we might be stuck.
        
    # Phase 3: Simulated Annealing
    sa_deadline = deadline - 15.0  # Leave 15s for final polish
    p = best_p.copy()
    current_cost = best_cost
    
    # Estimate steps
    remaining_time = sa_deadline - time.time()
    if remaining_time <= 0:
        return best_p.tolist()
        
    # Rough estimate of steps: delta_2opt is ~10-20us. Let's assume 100k steps/sec.
    est_steps = int(remaining_time * 80000)
    T = 500.0
    T_end = 0.1
    if est_steps > 100:
        alpha = (T_end / T) ** (1.0 / est_steps)
    else:
        alpha = 0.999
        
    # Use a random seed for reproducibility? No, let it be random.
    rng = np.random.RandomState(42)
    
    sa_steps = 0
    while time.time() < sa_deadline:
        i = rng.randint(n)
        j = rng.randint(n)
        if i == j:
            continue
            
        d = delta_2opt(p, i, j)
        
        if d < 0 or rng.rand() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= alpha
        if T < T_end:
            T = T_end
        sa_steps += 1
        
        # Periodic restart if stuck? 
        # If no improvement for a long time, maybe perturb?
        # For now, standard SA is robust.
        
    # Phase 4: Final Polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            best_j = -1
            best_d = 0
            for j in range(i+1, n):
                d = delta_2opt(p, i, j)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
                
    return best_p.tolist()
