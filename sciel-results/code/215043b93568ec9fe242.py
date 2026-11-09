import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            # Update in reverse to avoid using the same element multiple times in one step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def can_add(S, x):
        """
        Checks if x can be added to S.
        Returns True if valid, False otherwise.
        """
        n = len(S)
        if n == 0:
            return True
        
        # 1. Check if x is the mean of any subset of S
        # k * x = sum(subset of S), k >= 2
        dp_full = build_dp(S)
        max_sum = sum(S)
        
        for k in range(2, n + 1):
            t = k * x
            if t > max_sum:
                break
            if (dp_full[k] >> t) & 1:
                return False

        # 2. Check if any a in S becomes the mean of a subset of (S \ {a}) U {x}
        # The subset must include x.
        # Let the subset be {x} U T, where T is a subset of S \ {a} of size j.
        # Mean condition: a = (x + sum(T)) / (j + 1)
        # => (j + 1) * a = x + sum(T)
        # => sum(T) = (j + 1) * a - x
        # T must be a subset of S \ {a} of size j.
        # j >= 1 (since subset size >= 2, and one element is x)
        
        for a in S:
            # Build DP for S \ {a}
            S_minus_a = [y for y in S if y != a]
            dp_sub = build_dp(S_minus_a)
            max_sub = max_sum - a
            
            for j in range(1, n):
                t = (j + 1) * a - x
                if t < 0 or t > max_sub:
                    continue
                if (dp_sub[j] >> t) & 1:
                    return False
        
        return True

    def verify(S):
        """
        Independent verification of the set S.
        """
        S = list(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            sub = S[:i] + S[i + 1:]
            m = len(sub)
            if m < 2:
                continue
            dp = build_dp(sub)
            maxs = sum(sub)
            for k in range(2, m + 1):
                t = k * a
                if t > maxs:
                    break
                if (dp[k] >> t) & 1:
                    return False
        return True

    def conflict_count(x, S):
        """
        Count how many elements in S would conflict with x if added.
        A conflict exists if:
        1. x is the mean of some subset of S (size >= 2)
        2. Some a in S is the mean of a subset of (S \ {a}) U {x}
        """
        n = len(S)
        if n == 0:
            return 0
        
        conflicts = 0
        
        # Check if x is the mean of any subset of S
        dp_full = build_dp(S)
        max_sum = sum(S)
        
        for k in range(2, n + 1):
            t = k * x
            if t > max_sum:
                break
            if (dp_full[k] >> t) & 1:
                return 999  # High conflict score
        
        # Check if any a in S becomes the mean
        for a in S:
            S_minus_a = [y for y in S if y != a]
            dp_sub = build_dp(S_minus_a)
            max_sub = max_sum - a
            
            for j in range(1, n):
                t = (j + 1) * a - x
                if t < 0 or t > max_sub:
                    continue
                if (dp_sub[j] >> t) & 1:
                    conflicts += 1
        
        return conflicts

    def greedy_independent_set(candidates, deadline):
        """
        Greedy algorithm to build an independent set using conflict-aware ordering.
        """
        S = []
        Sset = set()
        remaining = list(candidates)
        
        while remaining and len(S) < MAXC:
            if time.time() > deadline:
                break
            
            # Find the candidate with minimum conflicts
            best_idx = -1
            best_conflicts = float('inf')
            
            for i, x in enumerate(remaining):
                c = conflict_count(x, S)
                if c < best_conflicts:
                    best_conflicts = c
                    best_idx = i
                if c >= 999:
                    continue
                # Early termination if we found a zero-conflict element
                if c == 0:
                    break
            
            if best_idx == -1 or best_conflicts >= 999:
                break
            
            x = remaining.pop(best_idx)
            S.append(x)
            Sset.add(x)
        
        return S

    best = []
    best_sz = 0

    # Strategy 1: Conflict-aware greedy with multiple orderings
    random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        if best_sz >= MAXC:
            break
        
        # Create candidate list with some randomization
        candidates = list(range(1, N + 1))
        
        # For first few trials, use structured orderings
        if trial == 0:
            # Ascending
            candidates = list(range(1, N + 1))
        elif trial == 1:
            # Descending
            candidates = list(range(N, 0, -1))
        elif trial == 2:
            # Even numbers first
            candidates = list(range(2, N + 1, 2)) + list(range(1, N + 1, 2))
        elif trial == 3:
            # Odd numbers first
            candidates = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        else:
            # Random shuffle
            random.shuffle(candidates)
        
        S = greedy_independent_set(candidates, deadline - 10)
        
        if len(S) > best_sz:
            if verify(S):
                best_sz = len(S)
                best = S[:]

    # Phase 2: Local Search (Swaps)
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        random.shuffle(unused)
        
        improved = True
        no_improve_counter = 0
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try to add
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if time.time() > deadline - 2:
                    break
                if can_add(S, u):
                    S.append(u)
                    Sset.add(u)
                    unused.remove(u)
                    improved = True
                    no_improve_counter = 0
                    break
            
            if len(S) >= MAXC:
                break
                
            if not improved:
                # Try swaps
                random.shuffle(unused)
                
                swapped = False
                max_swaps_to_try = 20 if time.time() > deadline - 10 else len(unused)
                
                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 2:
                        break
                    for i, s in enumerate(S):
                        # Try swapping s with u
                        newS = S[:i] + S[i + 1:]
                        if can_add(newS, u):
                            S = newS + [u]
                            Sset.remove(s)
                            Sset.add(u)
                            unused.remove(u)
                            unused.append(s)
                            improved = True
                            swapped = True
                            no_improve_counter = 0
                            break
                    if swapped:
                        break
                
                if not swapped:
                    no_improve_counter += 1
                    if no_improve_counter > 5:
                        break

        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety check
    if not verify(best):
        # Fallback: remove elements until valid
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} elements")
    print(res)
