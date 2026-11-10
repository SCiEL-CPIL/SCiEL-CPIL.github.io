"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    # Vertices 0..N-1 represent numbers 1..N
    # Build compatibility bitmasks: compat[i] has bit j set if lcm(i+1, j+1) <= N
    compat = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << j)
        compat[i] = mask
    
    # Greedy initialization: pick vertices in order of descending degree
    degrees = [bin(c).count('1') for c in compat]
    order = sorted(range(N), key=lambda i: -degrees[i])
    
    # Greedy set packing
    selected = set()
    for v in order:
        # Check if v is compatible with all already selected
        compatible = True
        for u in selected:
            if not (compat[v] >> u) & 1:
                compatible = False
                break
        if compatible:
            selected.add(v)
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local search: try swaps (remove one, add two compatible)
    # And single additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single additions
        for v in range(N):
            if v in best_set:
                continue
            # Check compatibility with all in best_set
            ok = True
            for u in best_set:
                if not (compat[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                best_set.add(v)
                improved = True
                break  # restart after addition
        
        if time.time() >= deadline:
            break
        
        # Try 1-for-2 swaps: remove u, add v1 and v2
        # For efficiency, iterate over candidates to remove
        for u in list(best_set):
            if time.time() >= deadline:
                break
            # Find all vertices compatible with (best_set - {u})
            remaining = best_set - {u}
            candidates = []
            for v in range(N):
                if v in remaining:
                    continue
                # v must be compatible with all in remaining
                ok = True
                for w in remaining:
                    if not (compat[v] >> w) & 1:
                        ok = False
                        break
                if ok:
                    candidates.append(v)
            
            # Try to find two candidates v1, v2 that are mutually compatible
            # and together with remaining form a valid set
            # Actually we just need v1 and v2 to be compatible with each other
            # and both compatible with remaining (already checked)
            for i_idx in range(len(candidates)):
                v1 = candidates[i_idx]
                for j_idx in range(i_idx + 1, len(candidates)):
                    v2 = candidates[j_idx]
                    if (compat[v1] >> v2) & 1:
                        # Valid swap
                        best_set.discard(u)
                        best_set.add(v1)
                        best_set.add(v2)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
        
        # Try 1-for-1 swaps for potential rearrangement (less likely to help but cheap)
        for u in list(best_set):
            if time.time() >= deadline:
                break
            remaining = best_set - {u}
            for v in range(N):
                if v in remaining:
                    continue
                ok = True
                for w in remaining:
                    if not (compat[v] >> w) & 1:
                        ok = False
                        break
                if ok:
                    # Also check v is compatible with u? No, we're replacing u.
                    # But we might want to keep size same or increase.
                    # 1-for-1 doesn't increase size. Skip unless it enables future additions.
                    # For now, skip pure 1-for-1.
                    pass
    
    # Convert back to list of numbers (1-indexed)
    result = sorted([v + 1 for v in best_set])
    
    # Final validation and safety
    if not result:
        return [1]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            g = gcd(a, b)
            if a * b // g > 512:
                # Should not happen, but fallback
                return list(range(1, 23))
    
    return result
# EVOLVE-BLOCK-END
