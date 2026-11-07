import time
import sys

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    # Strategy: For a given span S, the window [L, L+S] must contain at least one
    # multiple of each divisor, and we need to pick distinct ones.
    # Key insight: For small S, we can scan L efficiently.
    # For each L, the candidates for divisor d are multiples of d in [L, L+S].
    # The number of candidates is small (about S/d + 1).
    
    # We'll try S from 1 upward. For each S, we scan L.
    # To speed up: note that the set of "available multiples" only changes at
    # multiples of the divisors. But for small S, simple scanning is fine.
    
    # Better approach: For each S, instead of scanning all L, note that
    # a valid L must be such that for each divisor d, there's a multiple in [L, L+S].
    # This means L must be in the intersection over d of [m_d - S, m_d] for some multiple m_d of d.
    # Equivalently, L must be within S of some multiple of each d.
    
    # Even better: Let's just scan L but be smart about when to skip.
    # Actually, for S up to ~50, the scan is very fast.
    
    # Let's use a different structural approach:
    # Model as a constraint satisfaction / matching problem.
    # For a fixed window [L, L+S], build a bipartite graph:
    # Left side: divisors (10 nodes)
    # Right side: integers in [L, L+S] that are multiples of at least one divisor
    # Edge: divisor d connected to integer x if d divides x
    # We need a perfect matching (each divisor matched to a distinct integer).
    
    # For each S, we can scan L. But how many L to try?
    # The pattern of which multiples fall in the window repeats with period lcm(6,10,14,15,21,22,26,33,35,39).
    # That lcm is huge, so we can't rely on periodicity.
    
    # Instead: for each S, scan L from 1 to some bound. The bound can be large
    # but for small S, valid L's are dense enough that we find one quickly.
    
    # Let's try S from 1 to 200, and for each S, scan L up to 100000.
    # With early termination and fast matching, this should be feasible.
    
    # Use a fast matching check. For N=10, we can use backtracking with pruning.
    # Or use a bitset-based approach.
    
    # Actually, let's use a cleaner approach:
    # For each S, for each L (with some stepping), check feasibility via backtracking.
    
    # To reduce L search: note that if [L, L+S] works, then [L+k, L+k+S] might work
    # for some k. But the structure changes. Let's just scan L=1,2,3,...
    
    # Optimization: Precompute for each position x (up to some max), which divisors divide x.
    # But x can be up to 1000000, and we don't know L in advance.
    
    # Let's just do it directly. For small S, the inner loop is very fast.
    
    max_S = 200
    max_L = 200000  # Upper bound for L scan per S
    
    for S in range(1, max_S + 1):
        if time.time() > deadline:
            break
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            if L + S > MAXV:
                break
            
            # Build candidates for each divisor
            candidates = []
            valid = True
            for d in DIVISORS:
                k = (L + d - 1) // d
                m = k * d
                cands = []
                while m <= L + S:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking to find distinct assignment
            assignment = [None] * N
            used = set()
            
            # Sort by number of candidates for better pruning
            indices = sorted(range(N), key=lambda i: len(candidates[i]))
            
            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                result = [int(assignment[i]) for i in range(N)]
                return result

    # Fallback: return divisors themselves
    return DIVISORS[:]
