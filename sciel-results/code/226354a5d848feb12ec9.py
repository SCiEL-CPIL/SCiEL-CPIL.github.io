"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    # Denominators 1 to 10
    denoms = range(1, 11)
    # Precompute modular inverses
    invs = [pow(d, -1, 101) for d in denoms]
    
    # BFS on state space (residue, mask)
    # State: (current_residue, used_denominator_mask)
    # Goal: Find minimum popcount(mask) for each residue
    
    start_res = 0
    start_mask = 0
    
    q = deque()
    q.append((start_res, start_mask))
    visited = set()
    visited.add((start_res, start_mask))
    
    best_size = [101] * 101
    best_mask = [0] * 101
    is_covered = [False] * 101
    covered_count = 0
    
    while q:
        r, m = q.popleft()
        s = bin(m).count("1")
        
        # Update best if this is a smaller size for this residue
        # Must be non-empty (s > 0)
        if s > 0 and s < best_size[r]:
            best_size[r] = s
            best_mask[r] = m
            if not is_covered[r]:
                is_covered[r] = True
                covered_count += 1
                if covered_count == 101:
                    break
        
        # Expand to neighbors: add one more denominator
        for i in range(10):
            if (m >> i) & 1:
                continue
            new_m = m | (1 << i)
            new_r = (r + invs[i]) % 101
            if (new_r, new_m) not in visited:
                visited.add((new_r, new_m))
                q.append((new_r, new_m))
                
    # Construct the result lists
    representations = [None] * 101
    for r in range(101):
        m = best_mask[r]
        terms = [d for i, d in enumerate(denoms) if (m >> i) & 1]
        representations[r] = terms
        
    return representations
# EVOLVE-BLOCK-END
