"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def _quantum_rotation(a, b, p, theta, rng):
    """Quantum-inspired rotation step.
    
    Represents the permutation as a complex unitary matrix U where U[i, p[i]] = 1.
    The rotation operator is derived from the gradient of the QAP cost w.r.t. 
    the assignment. We compute the cost gradient for each facility-location pair,
    then apply a probabilistic rotation that allows tunneling through energy barriers.
    
    The key idea: instead of only accepting improvements, we use the quantum state
    to compute a probability distribution over potential moves that includes both
    improving and worsening moves, weighted by the quantum amplitude.
    """
    n = len(p)
    # Compute the cost gradient: for each facility i and location l, 
    # what is the marginal cost of moving facility i to location l?
    # grad[i, l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l] - current_contribution_i
    
    # Current contribution of facility i: sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
    # We'll compute the full gradient matrix efficiently
    
    # L[i, l] = sum_j a[i,j] * b[l, p[j]] (already computed in Benders)
    # R[i, l] = sum_j a[j,i] * b[p[j], l]
    
    # For now, use a simpler quantum-inspired approach:
    # Compute the "energy landscape" and apply a quantum rotation
    
    # The quantum state is represented by the permutation itself.
    # We compute a probability distribution over swap pairs based on:
    # 1. The cost delta (improving moves have higher probability)
    # 2. A quantum tunneling term that allows some worsening moves
    
    # Sample a few random pairs and compute their deltas
    candidates = []
    num_candidates = 10
    for _ in range(num_candidates):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            d = _swap_delta(a, b, p, i, j)
            candidates.append((i, j, d))
    
    if not candidates:
        return p
    
    # Compute quantum amplitudes: exp(-d / (2*theta)) for tunneling
    # This gives a smooth distribution that favors improvements but allows some worsening
    amplitudes = np.array([np.exp(-d / (2.0 * theta)) for _, _, d in candidates])
    amplitudes /= amplitudes.sum()
    
    # Select a move according to the quantum distribution
    idx = rng.choice(len(candidates), p=amplitudes)
    i, j, d = candidates[idx]
    
    # Apply the move with a quantum rotation probability
    # The rotation angle determines the probability of actually executing the move
    rotation_prob = np.abs(np.cos(theta * 0.5)) ** 2
    if rng.random() < rotation_prob:
        p[i], p[j] = p[j], p[i]
    
    return p


def _benders_cut(a, b, p, cut_coeff, cut_const, n):
    """Generate a Benders cut from current assignment p.
    
    The linearized surrogate cost is sum_i sum_j a[i,j] * b[p[i],p[j]].
    We maintain a lower-bound surrogate: for each facility i assigned to location l,
    the marginal contribution is sum_j a[i,j] * b[l, p[j]].
    
    A Benders cut says: for any assignment q, 
    cost(q) >= sum_i c_i(q[i]) + const
    where c_i(l) = sum_j a[i,j] * b[l, p[j]] (the linearized cost of putting i at l,
    given current assignment of others).
    
    We update cut coefficients to be the pointwise minimum over all cuts generated so far.
    """
    # Linearized cost matrix: L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # This is a @ b.T[p]  ... let's compute efficiently
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # = sum_j a[i,j] * b.T[p[j], l]
    # = (a @ b.T)[i, p[j]] summed over j... no
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Let's compute: for each i, L[i,:] = sum_j a[i,j] * b[:, p[j]]
    # = a[i,:] @ b[:, p[:]]  ... wait, that's sum_j a[i,j] * b[:,p[j]][l] which is what we want
    
    # L[i, l] = sum_j a[i,j] * b[l, p[j]]
    # Compute as: L = a @ b[:, p]  -- no, dimensions don't work directly
    # a is (n,n), b[:, p] is (n,n) where column j of b[:,p] is b[:,p[j]]
    # (a @ b[:,p])[i,l] = sum_j a[i,j] * b[:,p][j,l] = sum_j a[i,j] * b[j, p[j]][l]... no
    # b[:,p][j,l] = b[j, p[l]]... that's not right either.
    # b[:, p] means for each row index r, the column is p. So b[:,p][r,l] = b[r, p[l]].
    # We want b[l, p[j]]. That's b.T[p[j], l] or equivalently, if we let M = b.T, then M[p[j], l].
    # L[i,l] = sum_j a[i,j] * b[l, p[j]]
    # Let's just compute it: for each l, L[:,l] = sum_j a[:,j] * b[l, p[j]]
    # b[l, p[j]] for all j is b[l, p] which is a vector of length n.
    # So L[:,l] = a @ b[l, p]  (matrix-vector product)
    # L = a @ b[:, p].T ? No.
    # Let's do: L[i,l] = sum_j a[i,j] * b[l, p[j]]
    # Define V[l,j] = b[l, p[j]]. Then L[i,l] = sum_j a[i,j] * V[l,j] = (a @ V.T)[i,l]
    # V = b[:, p] has shape (n,n) where V[l,j] = b[l, p[j]]. Yes!
    # So L = a @ b[:, p].T
    # But b[:, p].T has shape (n,n) where element [j,l] = b[l, p[j]]. 
    # (a @ b[:,p].T)[i,l] = sum_j a[i,j] * b[:,p].T[j,l] = sum_j a[i,j] * b[l,p[j]]. Correct!
    
    L = a @ b[:, p].T  # L[i,l] = sum_j a[i,j] * b[l, p[j]]
    
    # Update cut: cut_coeff[i, l] = min(cut_coeff[i, l], L[i, l])
    # cut_const stays the same for this formulation (each cut is independent)
    np.minimum(cut_coeff, L, out=cut_coeff)
    
    return L


def _surrogate_cost(cut_coeff, p, n):
    """Compute surrogate cost: sum_i cut_coeff[i, p[i]]"""
    return int(np.sum(cut_coeff[np.arange(n), p]))


def _surrogate_assignment(cut_coeff, n, rng):
    """Assign facilities to locations using the surrogate cost matrix.
    Uses a greedy randomized assignment: process facilities in random order,
    assign each to its best available location per the surrogate."""
    assigned = np.zeros(n, dtype=bool)
    p = np.empty(n, dtype=np.int64)
    order = rng.permutation(n)
    for idx in order:
        # Find best unassigned location for facility idx
        row = cut_coeff[idx].copy()
        row[assigned] = np.inf
        best_loc = np.argmin(row)
        p[idx] = best_loc
        assigned[best_loc] = True
    return p


def _local_search_2opt(a, b, p, current_cost, deadline, max_passes=20):
    """First-improvement local search with swap moves."""
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 1:
            break
        improved = False
        for i in range(n):
            if time.time() >= deadline - 1:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return p, current_cost


def _quantum_annealing(a, b, p, current_cost, deadline, max_iters, T_start, T_min, cooling):
    """Quantum-inspired annealing that uses the quantum rotation operator.
    
    This function implements a quantum walk on the permutation space where:
    1. The state is represented by the permutation
    2. Transitions are governed by quantum amplitudes derived from the cost gradient
    3. The temperature controls the tunneling probability
    
    Returns the best permutation found and its cost.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    T = T_start
    
    for it in range(max_iters):
        if time.time() >= deadline - 5:
            break
        if T < T_min:
            break
        
        # Apply quantum rotation
        p_new = _quantum_rotation(a, b, p, T, rng_global)
        
        # Compute new cost
        new_cost = _qap_cost(a, b, p_new)
        
        # Accept if better, or with quantum probability if worse
        d = new_cost - current_cost
        if d < 0 or rng_global.random() < np.exp(-d / T):
            p = p_new
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling
    
    return best_p, best_cost


# Global RNG for quantum operations
rng_global = np.random.default_rng(42)


def solve(a, b):
    global rng_global
    n = len(a)
    deadline = time.time() + 115
    rng = np.random.default_rng(42)
    
    # Initial assignment: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    best_p = p0.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Benders cut storage: pointwise minimum of linearized cost matrices
    cut_coeff = np.full((n, n), np.inf, dtype=np.float64)
    
    # Generate initial cut from rank-matching solution
    _benders_cut(a, b, best_p, cut_coeff, 0, n)
    
    # Phase 1: Multi-restart SA with Benders-guided restarts (first 70% of time)
    T_start = 3000.0
    T_min = 0.5
    cooling = 0.99997
    max_iters_per_run = 8000
    cut_interval = 300
    
    phase1_deadline = deadline - 35
    
    while time.time() < phase1_deadline:
        # Choose starting point: either best solution or surrogate-guided
        if rng.random() < 0.3:
            # Use surrogate to generate a diverse starting point
            p = _surrogate_assignment(cut_coeff, n, rng)
        else:
            p = best_p.copy()
        
        current_cost = _qap_cost(a, b, p)
        T = T_start
        iters_since_cut = 0
        
        for it in range(max_iters_per_run):
            if time.time() >= phase1_deadline:
                break
            if T < T_min:
                break
            
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            d = _swap_delta(a, b, p, i, j)
            if d < 0 or rng.random() < np.exp(-d / T):
                p[i], p[j] = p[j], p[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    iters_since_cut = 0
            
            iters_since_cut += 1
            T *= cooling
            
            # Regenerate Benders cut from current best
            if iters_since_cut >= cut_interval:
                _benders_cut(a, b, best_p, cut_coeff, 0, n)
                iters_since_cut = 0
        
        # After each SA run, do a local search polish from best
        if time.time() < phase1_deadline - 5:
            p_ls = best_p.copy()
            cost_ls = best_cost
            p_ls, cost_ls = _local_search_2opt(a, b, p_ls, cost_ls, phase1_deadline)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
                # Update cut from improved solution
                _benders_cut(a, b, best_p, cut_coeff, 0, n)
    
    # Phase 2: Quantum-inspired annealing (next 20% of time)
    quantum_deadline = deadline - 12
    
    if time.time() < quantum_deadline:
        T_q_start = 500.0
        T_q_min = 1.0
        cooling_q = 0.99995
        max_q_iters = 5000
        
        p_q = best_p.copy()
        cost_q = best_cost
        
        p_q, cost_q = _quantum_annealing(a, b, p_q, cost_q, quantum_deadline, 
                                         max_q_iters, T_q_start, T_q_min, cooling_q)
        
        if cost_q < best_cost:
            best_cost = cost_q
            best_p = p_q.copy()
            _benders_cut(a, b, best_p, cut_coeff, 0, n)
        
        # Local search after quantum phase
        if time.time() < quantum_deadline - 3:
            p_ls = best_p.copy()
            cost_ls = best_cost
            p_ls, cost_ls = _local_search_2opt(a, b, p_ls, cost_ls, quantum_deadline)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Phase 3: Final high-precision polish: steepest descent with tighter tolerance
    p = best_p.copy()
    current_cost = best_cost
    max_passes = 25  # Increased from 15 for tighter polish
    for _ in range(max_passes):
        if time.time() >= deadline - 2:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() >= deadline - 2:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_d
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
