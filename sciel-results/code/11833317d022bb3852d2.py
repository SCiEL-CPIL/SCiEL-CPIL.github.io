import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    # Sort by weight (1/m) descending, so smallest m first
    moduli.sort(key=lambda m: 1.0/m, reverse=True)
    
    best_score = 0.0
    best_set = []
    
    # Precompute gcds? Or just use math.gcd on the fly.
    # For speed, we can precompute a compatibility matrix or just check on the fly.
    # Given N=180, on-the-fly is fine.
    
    def is_gcd_compatible(m, current_set):
        """Check if m can coexist with all moduli in current_set (gcd > 1)."""
        for m_i, _ in current_set:
            if math.gcd(m, m_i) == 1:
                return False
        return True

    def find_valid_residues(m, current_set):
        """Find all residues r in [0, m-1] that are disjoint from current_set."""
        valid_rs = []
        for r in range(m):
            is_valid = True
            for m_i, r_i in current_set:
                g = math.gcd(m, m_i)
                # Disjoint iff r != r_i mod g
                if r % g == r_i % g:
                    is_valid = False
                    break
            if is_valid:
                valid_rs.append(r)
        return valid_rs

    def dfs(idx, current_set, current_score):
        nonlocal best_score, best_set
        
        # Time check
        if time.time() - start_time > deadline:
            return

        # Pruning: Upper bound
        # Sum of weights of all remaining moduli that are gcd-compatible with current_set
        ub = current_score
        for j in range(idx, len(moduli)):
            m_j = moduli[j]
            if is_gcd_compatible(m_j, current_set):
                ub += 1.0 / m_j
        
        if ub <= best_score:
            return

        if idx == len(moduli):
            if current_score > best_score:
                best_score = current_score
                best_set = [list(x) for x in current_set]
            return

        m = moduli[idx]
        
        # Option 1: Skip m
        dfs(idx + 1, current_set, current_score)
        
        # Option 2: Include m
        if is_gcd_compatible(m, current_set):
            valid_rs = find_valid_residues(m, current_set)
            # To avoid excessive branching, if there are many valid residues,
            # we might just pick the first one? 
            # Actually, different residues have different impacts on future nodes.
            # But let's try all if count is small, otherwise just the first.
            if len(valid_rs) <= 5:
                for r in valid_rs:
                    new_set = current_set + [[m, r]]
                    dfs(idx + 1, new_set, current_score + 1.0/m)
            else:
                # Heuristic: try the first valid residue
                r = valid_rs[0]
                new_set = current_set + [[m, r]]
                dfs(idx + 1, new_set, current_score + 1.0/m)
                # Maybe try a few more?
                # For now, just one to keep speed up.

    dfs(0, [], 0.0)
    
    return best_set

