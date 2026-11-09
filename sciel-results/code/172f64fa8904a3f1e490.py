import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    N = 128
    
    def is_valid(S):
        """Check if set S (as set of ints) satisfies the constraint."""
        S_set = set(S)
        elems = sorted(S)
        n = len(elems)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    def check_valid_with_set(S_bool):
        """S_bool is a list of 129 booleans (index 0 unused). Check validity."""
        # Get the actual elements
        elems = [i for i in range(1, N+1) if S_bool[i]]
        n = len(elems)
        S_set = set(elems)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    def try_add(S_bool, x):
        """Try to add x to S_bool. Returns True if valid after adding."""
        if S_bool[x]:
            return True
        # Temporarily add
        S_bool[x] = True
        # Check all triples involving x
        elems = [i for i in range(1, N+1) if S_bool[i]]
        S_set = set(elems)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                for k in range(j+1, len(elems)):
                    a, b, c = elems[i], elems[j], elems[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        S_bool[x] = False
                        return False
        return True
    
    def remove(S_bool, x):
        S_bool[x] = False
    
    # Start with the seed: top half
    best_S = [False] * (N+1)
    for i in range(N//2+1, N+1):
        best_S[i] = True
    best_size = sum(best_S)
    
    # Local search: try to add elements from the bottom half
    # Then try swaps
    
    def local_search(S_bool, max_iter=50000):
        """Try to improve the set by adding, removing, or swapping."""
        current_size = sum(S_bool)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            
            # Try adding each element not in set
            for x in range(1, N+1):
                if not S_bool[x]:
                    if try_add(S_bool, x):
                        current_size += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swaps: remove one, add another
            in_set = [i for i in range(1, N+1) if S_bool[i]]
            not_in_set = [i for i in range(1, N+1) if not S_bool[i]]
            
            for out in in_set:
                if time.time() > deadline:
                    break
                S_bool[out] = False
                for inp in not_in_set:
                    if S_bool[inp]:
                        continue
                    if try_add(S_bool, inp):
                        improved = True
                        break
                    else:
                        S_bool[inp] = False
                if not improved:
                    S_bool[out] = True
        
        return sum(S_bool)
    
    # Run local search
    S = best_S[:]
    size = local_search(S)
    if size > best_size:
        best_size = size
        best_S = S[:]
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random initial set
        S = [False] * (N+1)
        # Start with a random subset
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        # Try to greedily add
        for x in candidates:
            if try_add(S, x):
                pass
        
        # Local search
        size = local_search(S)
        if size > best_size:
            best_size = size
            best_S = S[:]
    
    # Final result
    result = [i for i in range(1, N+1) if best_S[i]]
    return result

# Quick test
if __name__ == "__main__":
    r = construct()
    print(f"Size: {len(r)}")
    print(f"First few: {r[:10]}")
