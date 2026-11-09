"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import gcd
from functools import lru_cache
import itertools

def construct():
    MAX_M = 128
    
    # 1. Build the Non-Coprime Graph
    # Nodes are 2..128. Index 0 corresponds to modulus 2, index 1 to 3, etc.
    # Let's map modulus m to index m-2.
    N = MAX_M - 1
    indices = list(range(2, MAX_M + 1))
    # adj[i] is a bitset (int) where bit j is 1 if gcd(indices[i], indices[j]) > 1
    adj = [0] * N
    for i in range(N):
        m1 = indices[i]
        for j in range(i + 1, N):
            m2 = indices[j]
            if gcd(m1, m2) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. CSP Solver for Residue Assignment
    # Checks if a set of moduli can be assigned valid residues.
    # Returns True if solvable, False otherwise.
    def is_solvable(moduli):
        n = len(moduli)
        if n <= 1:
            return True
        
        # Precompute gcds
        gcds = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                gcds[i][j] = gcd(moduli[i], moduli[j])
                gcds[j][i] = gcds[i][j]
        
        # Domains: list of lists of possible residues
        # To reduce domain size, we can restrict residues to 0..g_max-1?
        # No, residues must be < m. But the constraint is mod g.
        # We can just use 0..m-1.
        
        # Use Backtracking with MRV
        # To make it faster, we can pre-check if any variable has an empty domain
        # initially (not possible here) or if a constraint is unsatisfiable locally.
        
        # Heuristic: Order variables by smallest domain (which is m) or most constrained?
        # Let's just use the order given, but we can sort by modulus size (smaller modulus = smaller domain).
        # Smaller modulus means fewer choices, so harder to satisfy?
        # Actually, smaller modulus means the constraint mod gcd is tighter?
        # Let's sort by modulus ascending.
        sorted_mods = sorted(moduli)
        # We need to map back to original indices if we wanted the assignment, but we only need bool.
        
        def backtrack(idx, assignment):
            if idx == n:
                return True
            
            m = sorted_mods[idx]
            # Try all residues 0..m-1
            # Optimization: We can skip residues that are equivalent mod gcd to already assigned neighbors?
            # We just iterate.
            for r in range(m):
                valid = True
                # Check against previously assigned variables
                for prev_idx in range(idx):
                    g = gcds[sorted_mods.index(sorted_mods[idx])][sorted_mods.index(sorted_mods[prev_idx])] # This is slow, use precomputed matrix on sorted indices?
                    # Better: Precompute a matrix of constraints based on the sorted order.
                    pass
        
        # Let's implement a cleaner CSP solver
        # Variables: 0..n-1 corresponding to sorted_mods
        # Constraints: (i, j, g)
        
        # Re-index for sorted_mods
        s_gcds = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s_gcds[i][j] = gcd(sorted_mods[i], sorted_mods[j])
        
        assignment = [-1] * n
        
        def check(idx, r):
            for prev in range(idx):
                if (r - assignment[prev]) % s_gcds[idx][prev] == 0:
                    return False
            return True
        
        def backtrack(idx):
            if idx == n:
                return True
            m = sorted_mods[idx]
            for r in range(m):
                if check(idx, r):
                    assignment[idx] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = -1
            return False
        
        return backtrack(0)

    # 3. Branch and Bound for Maximum Solvable Clique
    best_size = 0
    best_moduli = []
    
    # Order vertices for B&B
    # Heuristic: Higher degree first? Or specific ordering?
    # Let's try ordering by degree in the non-coprime graph.
    degrees = [bin(adj[i]).count("1") for i in range(N)]
    # Sort indices by degree descending
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    # We need to work with the original adjacency, but iterate in 'order'.
    # Actually, standard Max Clique B&B works on a specific vertex ordering.
    # Let's map the graph to the new order.
    # new_adj[i] is bitset in the new coordinate system.
    # This is complex. Let's just use the original indices but pick the next vertex from the candidates
    # based on the 'order' priority.
    
    # Simpler B&B:
    # candidates: bitset of available vertices
    # current_clique: list of vertices in current clique
    
    def popcount(x):
        return bin(x).count("1")
    
    def bnb(current_clique, candidates):
        nonlocal best_size, best_moduli
        
        # Upper bound: current size + number of candidates
        if len(current_clique) + popcount(candidates) <= best_size:
            return
        
        if candidates == 0:
            if len(current_clique) > best_size:
                # Verify solvability (should be true if we maintained it, but double check)
                mods = [indices[v] for v in current_clique]
                if is_solvable(mods):
                    best_size = len(current_clique)
                    best_moduli = mods[:]
            return
        
        # Select next vertex
        # We want to pick a vertex from candidates.
        # To prune effectively, we can iterate through candidates.
        # Get list of candidate vertices
        cand_list = []
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            cand_list.append(v)
            temp_cand ^= lsb
            
        # Sort candidates by some heuristic? 
        # Maybe just iterate.
        for v in cand_list:
            # If v is not in candidates, skip (should not happen)
            if not (candidates & (1 << v)):
                continue
                
            new_clique = current_clique + [v]
            
            # Check if new_clique is solvable
            mods = [indices[x] for x in new_clique]
            if not is_solvable(mods):
                continue
                
            # New candidates: intersection of current candidates and neighbors of v
            new_candidates = candidates & adj[v]
            
            bnb(new_clique, new_candidates)
            
            # Remove v from candidates for subsequent iterations in this loop
            # This is standard for Max Clique to avoid duplicate sets
            candidates ^= (1 << v)
            
            # Update upper bound check?
            # The check at the top of the function handles it for the next recursive call.
            # But we can also break early if len(new_clique) + popcount(new_candidates) <= best_size
            # This is handled by the recursive call's first line.

    # Initialize
    # Start with empty clique, all vertices as candidates
    initial_candidates = (1 << N) - 1
    bnb([], initial_candidates)
    
    # Construct result
    result = []
    # We need to find the actual residues for the best_moduli
    # Re-run CSP to get assignment?
    # is_solvable currently returns bool. We need to modify it to return the assignment or just re-solve.
    # Let's modify is_solvable to optionally return assignment.
    
    def solve_residues(moduli):
        n = len(moduli)
        if n == 0: return []
        if n == 1: return [[0, moduli[0]]]
        
        sorted_mods = sorted(moduli)
        # Map back
        s_gcds = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s_gcds[i][j] = gcd(sorted_mods[i], sorted_mods[j])
        
        assignment = [-1] * n
        
        def check(idx, r):
            for prev in range(idx):
                if (r - assignment[prev]) % s_gcds[idx][prev] == 0:
                    return False
            return True
        
        def backtrack(idx):
            if idx == n:
                return True
            m = sorted_mods[idx]
            for r in range(m):
                if check(idx, r):
                    assignment[idx] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = -1
            return False
        
        if not backtrack(0):
            return None # Should not happen if is_solvable said yes
            
        # Map back to original order? The problem doesn't require specific order, just a list of pairs.
        # But we need to pair residue with modulus.
        res = []
        for i in range(n):
            res.append([assignment[i], sorted_mods[i]])
        return res

    if best_moduli:
        res_pairs = solve_residues(best_moduli)
        if res_pairs:
            return res_pairs
        else:
            # Fallback to prime 11 strategy if solver failed (shouldn't)
            p = 11
            moduli = [p * k for k in range(1, 12)]
            return [[i, m] for i, m in enumerate(moduli)]
    else:
        # Fallback
        p = 11
        moduli = [p * k for k in range(1, 12)]
        return [[i, m] for i, m in enumerate(moduli)]

# EVOLVE-BLOCK-END
