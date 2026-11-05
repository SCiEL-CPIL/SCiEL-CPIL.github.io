import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Precompute divisibility table
    # divs[a][s] = True if a divides s
    max_sum = 2 * n
    divs = [[False] * (max_sum + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for multiple in range(a, max_sum + 1, a):
            divs[a][multiple] = True

    def can_add(x, S):
        """Check if x can be added to S without violating the property."""
        # 1. x is a: x < b <= c in S. Check x !| b+c.
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if divs[x][b + c]:
                    return False
        
        # 2. x is b: a < x <= c. a in S, c in S. Check a !| x+c.
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if divs[a][x + c]:
                    return False
        
        # 3. x is c: a < b <= x. a in S, b in S. Check a !| b+x.
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if divs[a][b + x]:
                    return False
        
        return True

    def get_blockers(r, S):
        """Find all elements in S that conflict with adding r."""
        blockers = set()
        
        # 1. r is a: pairs (b,c) in S with b>r, c>=b and r | b+c
        for b in S:
            if b <= r:
                continue
            for c in S:
                if c < b:
                    continue
                if divs[r][b + c]:
                    blockers.add(b)
                    blockers.add(c)
        
        # 2. r is b: pairs (a,c) in S with a<r, c>=r and a | r+c
        for a in S:
            if a >= r:
                continue
            for c in S:
                if c < r:
                    continue
                if divs[a][r + c]:
                    blockers.add(a)
                    blockers.add(c)
        
        # 3. r is c: pairs (a,b) in S with a<b<=r and a | b+r
        for a in S:
            if a >= r:
                continue
            for b in S:
                if b <= a or b > r:
                    continue
                if divs[a][b + r]:
                    blockers.add(a)
                    blockers.add(b)
        
        return blockers

    def greedy_construct(start_set=None):
        """Greedy construction from a starting set."""
        if start_set is not None:
            S = set(start_set)
        else:
            S = set(range(171, n + 1))
        
        candidates = list(range(170, 0, -1))
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x not in S and can_add(x, S):
                S.add(x)
        
        return S

    def local_search(S, R):
        """Local search with swap operations. Returns improved S and R."""
        improved = True
        # Use a temperature or just iterate
        while improved and time.time() < deadline - 8:
            improved = False
            # Shuffle R to avoid bias
            random.shuffle(R)
            for r in R:
                if time.time() > deadline - 8:
                    break
                
                if r in S:
                    continue
                    
                blockers = get_blockers(r, S)
                
                if not blockers:
                    S.add(r)
                    R.remove(r)
                    improved = True
                    break
                
                # Try each blocker removal for 1-for-2 swap
                found_swap = False
                for s in blockers:
                    if s not in S:
                        continue
                    
                    # Check if removing s allows adding r
                    if can_add(r, S - {s}):
                        # Try to also add another element from R
                        best_swap = None
                        best_r2 = None
                        
                        # Iterate R to find a second element
                        for r2 in R:
                            if r2 == r:
                                continue
                            if can_add(r2, (S - {s}) | {r}):
                                if best_swap is None:
                                    best_swap = (S - {s}) | {r} | {r2}
                                    best_r2 = r2
                                # Prefer swaps that add the smallest possible r2
                                # to maximize future opportunities
                                if r2 < (best_r2 if best_r2 is not None else float('inf')):
                                    best_swap = (S - {s}) | {r} | {r2}
                                    best_r2 = r2
                        
                        if best_swap is not None:
                            # Size increased by 1 (removed s, added r and r2)
                            S = best_swap
                            R.remove(r)
                            if best_r2 in R:
                                R.remove(best_r2)
                            improved = True
                            found_swap = True
                            break
                        
                        # If we couldn't find a 2nd element, consider a pure swap (1-for-1)
                        # This keeps size same but changes composition.
                        # Only do this if we are stuck (no 1-for-2 found in this pass?)
                        # For now, let's allow pure swaps if random condition met
                        if random.random() < 0.1: # 10% chance to accept pure swap
                             S = (S - {s}) | {r}
                             R.remove(r)
                             R.append(s)
                             improved = True
                             found_swap = True
                             break
                
                if improved:
                    break
        
        return S, R

    # Phase 1: Initial greedy construction
    S = greedy_construct()
    R = [x for x in range(1, n + 1) if x not in S]
    
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search
    S, R = local_search(S, R)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Phase 3: Multi-start with different initial sets
    random.seed(42)
    for trial in range(10): # Increased trials
        if time.time() > deadline - 10:
            break
        
        # Create a different starting point by randomly removing some elements
        # and then greedily rebuilding
        if len(S) > 10:
            # Remove a random subset to create diversity
            removals = random.sample(list(S), min(5, len(S) // 4))
            start_set = S - set(removals)
        else:
            start_set = set(range(171, n + 1))
        
        S_new = greedy_construct(start_set)
        R_new = [x for x in range(1, n + 1) if x not in S_new]
        
        S_new, R_new = local_search(S_new, R_new)
        
        if len(S_new) > best_size:
            best_S = set(S_new)
            best_size = len(S_new)
    
    return sorted(best_S)

# Test harness
if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
