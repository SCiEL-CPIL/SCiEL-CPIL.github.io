import time
import random
import sys

def _count_conflicts(S):
    """
    Counts the total number of subset-mean violations in S.
    A violation is (a, subset) where a is in S, subset is a subset of S\{a} with size >= 2,
    and mean(subset) == a.
    We use bitset DP for each element a.
    """
    n = len(S)
    if n < 3:
        return 0
    
    s_set = set(S)
    total_violations = 0
    
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        m = len(others)
        
        # dp[k] is a bitset where bit s is 1 if sum s is achievable with k elements
        # Python integers act as arbitrary precision bitsets
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        # Check for violations: sum = k * a for k >= 2
        for k in range(2, m + 1):
            target_sum = k * a
            if target_sum <= 512 * k: # Max possible sum
                if (dp[k] >> target_sum) & 1:
                    # There is at least one subset of size k with sum k*a
                    # How many? The problem says "No element may be the exact arithmetic mean of ANY subset".
                    # The verifier checks if ANY exists.
                    # The "score" for local search is the count of violations.
                    # If we just count 1 per (a, k) where a violation exists, it's a proxy.
                    # Better: count the number of subsets? That's hard to get from bitset directly without popcount of intersections, 
                    # but we can just count 1 per violating k for each a, or just count 1 per a if any k violates.
                    # Let's count 1 per (a, k) pair that has a violation. This encourages removing elements that are means of many different subset sizes.
                    total_violations += 1
                    
    return total_violations

def _is_valid(S):
    """
    Returns True if S has no subset-mean violations.
    """
    return _count_conflicts(S) == 0

def _has_3ap(S):
    """
    Fast check for 3-term AP (subset size 2).
    """
    s = set(S)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            a = S[i]
            b = S[j]
            c = 2 * b - a
            if c in s and c != a and c != b:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    rng = random.Random(42)
    
    best_set = [1]
    best_score = 0 # Maximize cardinality
    
    # Initial set of all numbers
    all_nums = list(range(1, 513))
    
    # Strategy: Randomized Local Search
    # State: A subset of all_nums
    # Move: Swap an element in S with an element not in S
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize with a random subset of moderate size
        # Start with a greedy construction to get a decent starting point
        # Or just random? Greedy is better for initialization.
        
        # Let's do a fast greedy init first
        current_S = []
        remaining = all_nums[:]
        rng.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            # Fast 3AP check
            if _has_3ap(new_S):
                continue
            # Full check
            if _is_valid(new_S):
                current_S = new_S
            if len(current_S) >= 48:
                break
                
        if len(current_S) > len(best_set):
            best_set = current_S[:]
            best_score = len(current_S)
            
        if best_score >= 48:
            break
            
        # Now perform local search swaps
        # We want to increase the size of the set.
        # Current size is K.
        # We can try to replace an element x in current_S with y not in current_S.
        # If the new set is valid, we keep it.
        # If it's not valid, we might still keep it if it has fewer conflicts?
        # The prompt suggests: "accepting the move if it reduces the total count of subset-mean violations".
        # But we also want to maximize cardinality.
        # A common approach: 
        # 1. Try to add an element. If valid, accept.
        # 2. If can't add, try a swap. If swap results in valid set, accept (size stays same, but we might be stuck in a local opt).
        # 3. If swap results in invalid set but fewer conflicts, accept? This allows "climbing" out of local minima.
        
        # Let's implement a mixed strategy:
        # 1. Attempt to add a random element from the complement.
        # 2. If that fails, attempt a swap.
        
        # To speed up, we maintain a set of candidates for addition.
        # And a set of elements to remove.
        
        S_set = set(current_S)
        complement = [x for x in all_nums if x not in S_set]
        
        no_improve_count = 0
        current_conflicts = _count_conflicts(current_S)
        
        for step in range(5000): # Max steps per restart
            if time.time() > deadline:
                break
                
            if len(current_S) >= 48:
                break
                
            # 1. Try to add
            if complement:
                y = rng.choice(complement)
                new_S = current_S + [y]
                if _has_3ap(new_S):
                    # Fast reject
                    pass
                elif _is_valid(new_S):
                    # Accept
                    current_S = new_S
                    S_set.add(y)
                    complement.remove(y)
                    no_improve_count = 0
                    if len(current_S) > len(best_set):
                        best_set = current_S[:]
                        best_score = len(current_S)
                    continue
                else:
                    # Invalid, try swap?
                    pass
            
            # 2. Try swap
            if not current_S or not complement:
                break
                
            x = rng.choice(current_S)
            y = rng.choice(complement)
            
            if x == y:
                continue
                
            new_S = [z for z in current_S if z != x] + [y]
            
            # Check validity
            if _has_3ap(new_S):
                # Fast reject for 3AP
                # We can check conflict count roughly?
                # Or just skip if 3AP exists?
                # 3AP is a type of conflict.
                # Let's compute full conflict count to guide the search.
                pass
            
            new_conflicts = _count_conflicts(new_S)
            
            # Accept if:
            # a) It is valid (conflicts == 0)
            # b) It has fewer conflicts than current (hill climbing on conflicts)
            # c) Random restart/probability?
            
            if new_conflicts == 0:
                # Valid swap
                current_S = new_S
                S_set.remove(x)
                S_set.add(y)
                complement.remove(y)
                complement.append(x)
                no_improve_count = 0
                current_conflicts = 0
                # Size is unchanged, but we might be in a better position to add later
            elif new_conflicts < current_conflicts:
                # Move towards validity
                current_S = new_S
                S_set.remove(x)
                S_set.add(y)
                complement.remove(y)
                complement.append(x)
                current_conflicts = new_conflicts
                no_improve_count = 0
            else:
                # No improvement in conflicts
                no_improve_count += 1
                if no_improve_count > 100:
                    # Break out of this local minimum, restart search phase for this restart
                    break

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    if len(s) > 0:
        print(f"Valid: {_is_valid(s)}")
