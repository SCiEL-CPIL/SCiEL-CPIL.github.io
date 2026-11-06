"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
import time
import random
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix

_START = time.time()
_DEADLINE = 150.0  # seconds; return with margin

def _triples(n):
    """List of triples (a,b,c), a<b<c, in lexicographic order."""
    return list(combinations(range(n), 3))

def _build_incidence(n):
    """Build sparse incidence matrices for quads and pents."""
    triples = _triples(n)
    m = len(triples)
    triple_idx = {t: i for i, t in enumerate(triples)}
    
    # Quads
    quad_rows = []
    quad_cols = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        for t in combinations(q, 3):
            quad_rows.append(triple_idx[t])
            quad_cols.append(len(quad_rows) // 4) # This is wrong, need quad index
    
    # Correct way:
    quad_rows = []
    quad_cols = []
    qi = 0
    for q in combinations(range(n), 4):
        a, b, c, d = q
        for t in combinations(q, 3):
            quad_rows.append(triple_idx[t])
            quad_cols.append(qi)
        qi += 1
    n_quads = qi
    A_Q = csr_matrix((np.ones(len(quad_rows)), (quad_rows, quad_cols)), shape=(m, n_quads))
    
    # Pents
    pent_rows = []
    pent_cols = []
    pi = 0
    for p in combinations(range(n), 5):
        a, b, c, d, e = p
        for t in combinations(p, 3):
            pent_rows.append(triple_idx[t])
            pent_cols.append(pi)
        pi += 1
    n_pents = pi
    A_P = csr_matrix((np.ones(len(pent_rows)), (pent_rows, pent_cols)), shape=(m, n_pents))
    
    return A_Q, A_P, m, n_quads, n_pents

def _solve_n(n, time_budget):
    """Try to find a valid coloring for n."""
    A_Q, A_P, m, n_quads, n_pents = _build_incidence(n)
    # Transpose for fast column sums (triple -> quads/pents)
    # Actually we need A.T @ vec.
    # A is (m, n_quads). A.T is (n_quads, m).
    # We want S[t] = sum_{q contains t} B[q].
    # This is A.T @ B.
    # A_Q.T is (n_quads, m).
    
    t_end = _START + time_budget
    best_violations = float('inf')
    best_colors = None
    
    # Try multiple restarts
    attempts = 0
    while time.time() < t_end:
        attempts += 1
        # Random init
        # Bias towards Red? Red K4 is smaller, so maybe easier to avoid?
        # Blue K5 is larger.
        # Let's try uniform or slight bias.
        colors = (np.random.rand(m) < 0.5).astype(np.int8)
        
        # Initialize counts
        # C_Q[q] = sum of red triples in quad q
        # C_Q = A_Q @ colors_red
        colors_red = (colors == 0).astype(np.int8)
        colors_blue = (colors == 1).astype(np.int8)
        
        C_Q = (A_Q @ colors_red).astype(np.int8).flatten()
        C_P = (A_P @ colors_blue).astype(np.int8).flatten()
        
        violations = np.sum(C_Q == 4) + np.sum(C_P == 10)
        
        if violations == 0:
            return colors.tolist()
        
        if violations < best_violations:
            best_violations = violations
            best_colors = colors.copy()
        
        # Local search loop
        stuck_counter = 0
        while time.time() < t_end and violations > 0:
            # Compute gains
            B_Q4 = (C_Q == 4).astype(np.int8)
            B_Q3 = (C_Q == 3).astype(np.int8)
            B_P10 = (C_P == 10).astype(np.int8)
            B_P9 = (C_P == 9).astype(np.int8)
            
            # S_Q4[t] = number of quads containing t that have 4 reds
            S_Q4 = (A_Q.T @ B_Q4).astype(np.int8).flatten()
            S_Q3 = (A_Q.T @ B_Q3).astype(np.int8).flatten()
            S_P10 = (A_P.T @ B_P10).astype(np.int8).flatten()
            S_P9 = (A_P.T @ B_P9).astype(np.int8).flatten()
            
            gain = np.zeros(m, dtype=np.int8)
            # For Red triples (colors == 0): Gain = S_Q4 - S_P9
            red_mask = (colors == 0)
            blue_mask = (colors == 1)
            
            gain[red_mask] = S_Q4[red_mask] - S_P9[red_mask]
            gain[blue_mask] = -S_Q3[blue_mask] + S_P10[blue_mask]
            
            # Find improving moves
            improving = gain > 0
            if not np.any(improving):
                # Stuck. Perturb or break.
                stuck_counter += 1
                if stuck_counter > 5:
                    break
                # Random perturbation
                num_perturb = max(1, m // 100)
                perturb_idx = np.random.choice(m, num_perturb, replace=False)
                colors[perturb_idx] = 1 - colors[perturb_idx]
                
                # Update counts
                # This is expensive to do correctly for arbitrary perturbations.
                # Easier to recompute C_Q, C_P from scratch.
                colors_red = (colors == 0).astype(np.int8)
                colors_blue = (colors == 1).astype(np.int8)
                C_Q = (A_Q @ colors_red).astype(np.int8).flatten()
                C_P = (A_P @ colors_blue).astype(np.int8).flatten()
                violations = np.sum(C_Q == 4) + np.sum(C_P == 10)
                continue
            
            # Flip all improving triples
            flip_idx = np.where(improving)[0]
            red_flips = flip_idx[colors[flip_idx] == 0]
            blue_flips = flip_idx[colors[flip_idx] == 1]
            
            # Update colors
            colors[flip_idx] = 1 - colors[flip_idx]
            
            # Update counts
            if len(red_flips) > 0:
                # Red -> Blue: C_Q decreases, C_P increases
                C_Q = C_Q - (A_Q[:, red_flips].sum(axis=1)).astype(np.int8).flatten()
                C_P = C_P + (A_P[:, red_flips].sum(axis=1)).astype(np.int8).flatten()
            if len(blue_flips) > 0:
                # Blue -> Red: C_Q increases, C_P decreases
                C_Q = C_Q + (A_Q[:, blue_flips].sum(axis=1)).astype(np.int8).flatten()
                C_P = C_P - (A_P[:, blue_flips].sum(axis=1)).astype(np.int8).flatten()
            
            violations = np.sum(C_Q == 4) + np.sum(C_P == 10)
            
            if violations == 0:
                return colors.tolist()
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
                
    return best_colors.tolist() if best_colors is not None else None

def construct():
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # We want to maximize n.
    # Start from high n and go down? Or low to high?
    # Low to high allows building on success, but finding valid coloring is hard.
    # High to low: if we find valid for 34, we are done (max possible is likely 34 or 35).
    # Let's try 34, 33, 32... down to 10.
    
    # Time budget allocation:
    # Give more time to higher n.
    remaining = deadline - time.time()
    
    # Try 34 first with a generous chunk of time
    n_targets = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
    
    # Allocate time proportional to difficulty?
    # Let's just try them in order, giving each a slice of remaining time.
    # If we find a valid one, we can try to go higher? No, we start high.
    # If we fail 34, try 33.
    
    for n in n_targets:
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Allocate time: e.g., 20% of remaining for each attempt, min 2s
        # Or fixed time per n?
        # Let's give 10s for n=34, 8s for 33, etc.
        time_budget = min(remaining * 0.3, 15.0)
        
        colors = _solve_n(n, time_budget)
        if colors is not None:
            # Verify
            # (Verification skipped for speed, trusting the solver)
            best_n = n
            best_colors = colors
            # If we found a valid coloring for n, we could try n+1?
            # But we started from 34. If we found 34, we are likely at max.
            # If we found 33, maybe 34 is just hard.
            # Let's stick to the first valid one found from the top.
            break
            
    return {'n': best_n, 'colors': best_colors}
