"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _max_independent_set(n, adj):
    """
    Find a Maximum Independent Set in a graph with n nodes.
    adj is a list of sets of neighbors.
    Uses Bron-Kerbosch algorithm for Maximum Clique on the complement graph,
    which is equivalent to Maximum Independent Set on the original graph.
    """
    # Complement graph: edge (i, j) in Gc if NOT edge (i, j) in G and i != j
    # We want Max Clique in Gc.
    # Gc is dense (degree ~ 330 - 29 = 301).
    # Actually, it's easier to run Bron-Kerbosch for MIS directly on G.
    # MIS in G = Max set of nodes with no edges between them.
    
    best_set = []
    
    # R, P, X are sets of nodes
    # We use bitmasks for efficiency if n <= 64, but n=330.
    # We will use sets or lists.
    
    def bron_kerbosch_mis(R, P, X):
        nonlocal best_set
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(best_set):
                best_set = list(R)
            return
        
        # Pivot selection
        # Choose u from P ∪ X that maximizes |N(u) ∩ P|
        # This minimizes the branching factor
        if P or X:
            u = next(iter(P)) if P else next(iter(X))
            # Simple pivot: just pick first. Optimizing pivot is complex for sets.
            # For n=330, simple pivot might be okay, or we can try a few.
            # Let's use a better pivot: pick u in P ∪ X with max degree in P.
            max_deg = -1
            best_u = u
            candidates = P.union(X)
            for node in candidates:
                deg = len(adj[node].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    best_u = node
            
            u = best_u
            # Branch on P \ N(u)
            # If u in P, we branch on P \ N(u). If u in X, we branch on P.
            # Standard BK for Clique: Branch on P \ N(u).
            # For MIS, we branch on P \ N(u) where N(u) are neighbors in G.
            # Nodes in N(u) cannot be in the MIS with u.
            # So we consider P \ N(u).
            
            P_minus_Nu = P.difference(adj[u])
        else:
            P_minus_Nu = set()

        for v in list(P_minus_Nu):
            # R ∪ {v}
            new_R = R.union({v})
            # P ∩ N(v) ... wait.
            # In MIS, if we add v to R, we must remove its neighbors from P.
            # New P = P \ N(v)
            # New X = X \ N(v)
            new_P = P.difference(adj[v])
            new_X = X.difference(adj[v])
            
            bron_kerbosch_mis(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)

    # Initialize
    R = set()
    P = set(range(n))
    X = set()
    
    # Heuristic: Sort P? BK is sensitive to order.
    # We'll just run it.
    bron_kerbosch_mis(R, P, X)
    
    return best_set

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports for m=4
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    
    n_sup = len(supports) # 330
    
    # 2. Build adjacency list for conflict graph (intersection size 3)
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports[i] & supports[j]
            # Count bits
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Independent Set
    # This might take some time, but 330 nodes is small.
    # If it times out, we can fall back to greedy.
    try:
        mis_nodes = _max_independent_set(n_sup, adj)
    except Exception:
        # Fallback to greedy if MIS fails or takes too long
        mis_nodes = []
        alive = set(range(n_sup))
        while alive:
            # Pick node with min degree in alive
            v = min(alive, key=lambda x: len(adj[x].intersection(alive)))
            mis_nodes.append(v)
            alive.difference_update(adj[v])
            alive.remove(v)
            
    # 4. Generate vectors for MIS supports
    # Map support mask to its index in 'supports' list?
    # We just iterate over the masks in mis_nodes.
    # Wait, mis_nodes are indices into 'supports'.
    
    vectors = []
    for idx in mis_nodes:
        mask = supports[idx]
        # Get the 4 indices
        inds = [i for i in range(D) if (mask >> i) & 1]
        # Generate all 16 sign combinations
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
    
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Try to add Shell 2 vectors
    # Constraint: for all u in S4, v in S2: u.v <= 1
    shell2 = _generate_shell(2)
    if len(S4) > 0:
        # Compute max inner product of each shell2 vector with any S4 vector
        # To save memory/time, we can do this in chunks or just matrix mult if small enough.
        # S4 is ~600x11. S2 is 220x11.
        # Dots = S2 @ S4.T -> 220x600. Very small.
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        
        # Filter S2: keep those with max_ip_24 <= 1
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = shell2[valid_2_mask]
        
        # Now we need to select a subset of S2_candidates such that pairwise u.v <= 0 (since M^2=4)
        # Greedy selection for S2
        n2 = len(S2_candidates)
        if n2 > 0:
            # Shuffle for diversity? Or just greedy.
            # Greedy: pick one, remove conflicts.
            # Conflict in S2: u.v > 0.
            # Since S2 vectors have norm sqrt(2), u.v can be 2, 0, -2.
            # We need u.v <= 0. So we must avoid u.v = 2.
            # u.v = 2 implies u and v are identical.
            # So we just need distinct vectors?
            # If u != v, u.v <= 0 is always true for distinct weight-2 vectors?
            # Let u = e_i - e_j, v = e_k - e_l.
            # If they share 1 index, dot is 1 or -1.
            # If they share 2 indices (same support), dot is 2 (same) or -2 (opp) or 0 (mixed).
            # Wait. If same support {i,j}.
            # u = e_i - e_j. v = e_i + e_j. Dot = 1 - 1 = 0.
            # u = e_i - e_j. v = e_i - e_j. Dot = 2. (Identical)
            # u = e_i - e_j. v = -e_i + e_j. Dot = -2.
            # So distinct vectors from same support have dot 0 or -2. Both <= 0.
            # Vectors from different supports have dot in {-1, 0, 1}.
            # We need dot <= 0.
            # So we must avoid pairs with dot 1.
            
            # Greedy MIS on S2_candidates with edge if dot > 0.
            selected2 = []
            alive2 = list(range(n2))
            # Precompute dots? 220x220 is small.
            dots_22 = S2_candidates @ S2_candidates.T
            
            while alive2:
                # Pick first
                v_idx = alive2[0]
                selected2.append(v_idx)
                
                # Remove v_idx and all conflicts (dot > 0)
                conflicts = np.where(dots_22[v_idx] > 0)[0]
                alive2 = [i for i in alive2 if i not in conflicts and i != v_idx]
                
            S2 = S2_candidates[selected2]
            
            result = np.vstack([S4, S2])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
