import time
import random
from functools import lru_cache

def construct():
    """
    Return up to 48 distinct positive integers in [1, 512] such that no element is the
    arithmetic mean of any subset of at least two other elements.
    Maximize cardinality.
    """
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
    MAX_N = 512
    TARGET = 48

    def is_valid_set(S):
        """Check if set S is valid: no element is mean of any subset of >=2 others."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            x = S_list[i]
            # Check if x is mean of any subset of S_list \ {x} with size >= 2
            # Equivalent: exists subset T of S_list\{x}, |T|>=2, sum(T) = |T|*x
            # Let d_j = S_list[j] - x for j != i
            # Need non-empty subset of d's summing to 0 (size>=2 since no d=0)
            diffs = [S_list[j] - x for j in range(n) if j != i]
            if not diffs:
                continue
            min_d = sum(d for d in diffs if d < 0)
            max_d = sum(d for d in diffs if d > 0)
            if min_d == 0 and max_d == 0:
                continue
            offset = -min_d
            # Bitset DP for non-empty subsets only
            bits = 0
            for d in diffs:
                if d >= 0:
                    bits |= (bits << d) | (1 << (offset + d))
                else:
                    bits |= (bits >> (-d)) | (1 << (offset + d))
            if (bits >> offset) & 1:
                return False
        return True

    def try_add(S, x):
        """Try to add x to set S. Returns True if S ∪ {x} is valid."""
        S_list = sorted(S)
        n = len(S_list)
        
        # Check 1: Is x mean of any subset of S with size >= 2?
        if n >= 2:
            diffs = [y - x for y in S_list]
            min_d = sum(d for d in diffs if d < 0)
            max_d = sum(d for d in diffs if d > 0)
            if not (min_d == 0 and max_d == 0):
                offset = -min_d
                bits = 0
                for d in diffs:
                    if d >= 0:
                        bits |= (bits << d) | (1 << (offset + d))
                    else:
                        bits |= (bits >> (-d)) | (1 << (offset + d))
                if (bits >> offset) & 1:
                    return False
        
        # Check 2: For each y in S, is y mean of any subset of (S \ {y}) ∪ {x} with size >= 2?
        for y in S_list:
            others = [z for z in S_list if z != y] + [x]
            if len(others) < 2:
                continue
            diffs = [w - y for w in others]
            min_d = sum(d for d in diffs if d < 0)
            max_d = sum(d for d in diffs if d > 0)
            if not (min_d == 0 and max_d == 0):
                offset = -min_d
                bits = 0
                for d in diffs:
                    if d >= 0:
                        bits |= (bits << d) | (1 << (offset + d))
                    else:
                        bits |= (bits >> (-d)) | (1 << (offset + d))
                if (bits >> offset) & 1:
                    return False
        
        return True

    # Greedy with smart ordering
    def greedy_construct():
        selected = set()
        available = set(range(1, MAX_N + 1))
        
        # Start with well-spaced numbers to minimize conflicts
        # Use a deterministic good seed: powers of 2 give good separation
        # But we need up to 48, so let's be smarter
        
        for _ in range(TARGET):
            if time.time() > deadline:
                break
            
            # Score each candidate by how "conflict-free" it is
            # Prefer numbers that are far from existing elements (reduces mean conflicts)
            if not selected:
                # First element: pick something central
                x = random.choice([128, 256, 384])
                selected.add(x)
                available.remove(x)
                continue
            
            # For each candidate, estimate conflict potential
            # Simple heuristic: prefer numbers that don't create arithmetic progressions
            candidates = list(available)
            random.shuffle(candidates)
            
            # Try candidates in random order, accept first valid one
            # But also try some "smart" candidates: numbers far from existing ones
            best_x = None
            
            # First pass: try random candidates
            for x in candidates[:min(50, len(candidates))]:
                if try_add(selected, x):
                    best_x = x
                    break
            
            if best_x is None:
                # Second pass: try all remaining
                for x in candidates:
                    if try_add(selected, x):
                        best_x = x
                        break
            
            if best_x is None:
                # Try to find any valid addition by checking more carefully
                found = False
                for x in sorted(available, key=lambda v: (min(abs(v - s) for s in selected))):
                    if try_add(selected, x):
                        best_x = x
                        found = True
                        break
                if not found:
                    break
            
            selected.add(best_x)
            available.remove(best_x)
        
        return selected

    # Simulated Annealing
    def simulated_annealing(initial, time_budget):
        current = set(initial)
        best = set(current)
        best_size = len(current)
        
        T = 1.0  # Temperature
        T_min = 0.01
        cooling = 0.995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > T_min:
            if not current:
                break
            
            # Generate neighbor: swap one element or add/remove
            op = random.random()
            
            if op < 0.7 and len(current) > 1:
                # Swap
                used = list(current)
                i = random.randint(0, len(used) - 1)
                available = set(range(1, MAX_N + 1)) - current
                if not available:
                    continue
                new_val = random.choice(list(available))
                old_val = used[i]
                
                candidate = set(current)
                candidate.remove(old_val)
                candidate.add(new_val)
                
                if is_valid_set(candidate):
                    current = candidate
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                elif random.random() < T:
                    # Accept worse solution with probability T
                    current = candidate
            elif op < 0.85 and len(current) < TARGET:
                # Add
                available = set(range(1, MAX_N + 1)) - current
                if not available:
                    continue
                new_val = random.choice(list(available))
                candidate = set(current)
                candidate.add(new_val)
                
                if is_valid_set(candidate):
                    current = candidate
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                elif random.random() < T:
                    current = candidate
            else:
                # Remove
                if len(current) > 1:
                    to_remove = random.choice(list(current))
                    candidate = set(current)
                    candidate.remove(to_remove)
                    current = candidate
            
            T *= cooling
        
        return best

    # Main strategy
    # Try multiple random restarts with greedy + SA
    best_result = set()
    
    # First: try a structured greedy
    result = greedy_construct()
    if len(result) > len(best_result):
        best_result = set(result)
    
    # Use remaining time for simulated annealing from best result
    elapsed = time.time() - start
    remaining = deadline - time.time()
    if remaining > 5:
        sa_result = simulated_annealing(list(best_result), remaining - 3)
        if len(sa_result) > len(best_result):
            best_result = set(sa_result)
    
    # Final extension attempt
    if len(best_result) < TARGET and time.time() < deadline - 2:
        available = set(range(1, MAX_N + 1)) - best_result
        while time.time() < deadline - 1 and len(best_result) < TARGET:
            found = False
            candidates = list(available)
            random.shuffle(candidates)
            for x in candidates:
                if try_add(best_result, x):
                    best_result.add(x)
                    available.remove(x)
                    found = True
                    break
            if not found:
                break
    
    return sorted(best_result)

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
