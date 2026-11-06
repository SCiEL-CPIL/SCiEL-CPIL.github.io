import random
import time
import sys

def construct():
    """
    Constructs a set of up to 48 distinct positive integers <= 512.
    No element is the mean of any subset of at least two other elements.
    Uses Greedy Initialization followed by Local Search with Repair.
    """
    N = 512
    MAX_SIZE = 48
    
    # Fast DP verification
    def is_valid_and_count(S):
        """
        Returns (is_valid, violation_count).
        Uses integer bitset DP.
        """
        S_list = list(S)
        m = len(S_list)
        if m < 3:
            return True, 0
        
        total_violations = 0
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            # DP for others
            # dp[k] is a bitset where bit s is 1 if sum s is possible with k elements
            max_sum = sum(others)
            dp = [0] * (m_others + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            # Check violations for a
            for k in range(2, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    total_violations += 1
                    return False, 1
        
        return True, 0

    def get_violation(S):
        """
        Returns (a, T) if a is the mean of subset T of S\{a}, else (None, None).
        T is a list of elements.
        """
        S_list = list(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            m_others = len(others)
            if m_others < 2:
                continue
            
            max_sum = sum(others)
            
            # Check k=2
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if others[i] + others[j] == 2 * a:
                        return a, [others[i], others[j]]
            
            # Check k=3
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    rem = 3 * a - others[i] - others[j]
                    if rem > others[j] and rem in others:
                        if rem != others[i] and rem != others[j]:
                            return a, [others[i], others[j], rem]
            
            # Fallback: Use DP to check existence for k>=4
            dp = [0] * (m_others + 1)
            dp[0] = 1
            for x in others:
                for k in range(m_others, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            
            for k in range(4, m_others + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return a, None
        
        return None, None

    def greedy_construct(order):
        """
        Builds a set by adding elements in the given order if valid.
        """
        S = set()
        for x in order:
            if len(S) >= MAX_SIZE:
                break
            S_trial = S.copy()
            S_trial.add(x)
            is_valid, _ = is_valid_and_count(S_trial)
            if is_valid:
                S = S_trial
        return S

    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    candidates = list(range(1, N + 1))
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Greedy Initialization with multiple orders
    # 1. Descending
    S_desc = greedy_construct(range(N, 0, -1))
    if len(S_desc) > best_size:
        best_set = S_desc
        best_size = len(S_desc)
        
    # 2. Ascending
    S_asc = greedy_construct(range(1, N + 1))
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        
    # 3. Random Orders (a few)
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        random.shuffle(candidates)
        S_rand = greedy_construct(candidates)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
    # Phase 2: Local Search / Repair / Expansion
    # Start with the best greedy set
    S = best_set.copy()
    
    # If we didn't reach MAX_SIZE, try to expand/swap
    # We will use the repair logic from the original but applied to this better start
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # If we have a valid set, try to improve it
        # If current S is valid, try to add/swap
        # If not valid, repair it
        
        # Check if current S is valid
        is_valid, _ = is_valid_and_count(S)
        
        if not is_valid:
            # Repair S
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                is_valid, _ = is_valid_and_count(S)
                if is_valid:
                    break
                a, T = get_violation(S)
                if a is None:
                    break
                r = random.random()
                if T is not None and len(T) > 0 and r < 0.5:
                    victim = random.choice(T)
                    S.remove(victim)
                elif r < 0.8:
                    S.remove(a)
                else:
                    others = [x for x in S if x != a]
                    if others:
                        S.remove(random.choice(others))
                    else:
                        S.remove(a)
        
        # Now S is valid (or we gave up on repair)
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            
        if len(S) == MAX_SIZE:
            is_valid, _ = is_valid_and_count(S)
            if is_valid:
                return list(S)
        
        # Try to expand/swap to reach MAX_SIZE
        # Use a more aggressive local search on the valid set S
        
        for step in range(200):
            if len(S) >= MAX_SIZE or time.time() - start_time > time_limit:
                break
            
            remaining = [x for x in candidates if x not in S]
            if not remaining:
                break
                
            # Pick a random candidate to add
            c = random.choice(remaining)
            S_trial = S.copy()
            S_trial.add(c)
            
            is_valid_trial, _ = is_valid_and_count(S_trial)
            
            if is_valid_trial:
                S = S_trial
                if len(S) > best_size:
                    best_set = S.copy()
                    best_size = len(S)
                # If we reached max size, check and return
                if len(S) == MAX_SIZE:
                    is_valid_final, _ = is_valid_and_count(S)
                    if is_valid_final:
                        return list(S)
            else:
                # Try to fix S_trial by removing an element from S (swap)
                # Find violation in S_trial
                a, T = get_violation(S_trial)
                if a is not None:
                    # If we can remove an element from S (not c) to fix it, we swap
                    # This keeps size same, but changes composition
                    # We want to increase size, so swaps are only useful if they enable future additions
                    # Or if we are stuck at size < MAX_SIZE, we might need to swap to a "better" set that can grow.
                    
                    # Simple strategy: if T contains elements from S, remove one.
                    # If a is in S, remove a.
                    # If a is c, removing c reverts to S (no change).
                    
                    candidates_to_remove = []
                    if T:
                        candidates_to_remove.extend([x for x in T if x in S])
                    if a in S and a != c:
                        candidates_to_remove.append(a)
                    
                    if candidates_to_remove:
                        victim = random.choice(candidates_to_remove)
                        S_trial.remove(victim)
                        is_valid_swap, _ = is_valid_and_count(S_trial)
                        if is_valid_swap:
                            # Accept swap with some probability to explore
                            if random.random() < 0.3:
                                S = S_trial
                                if len(S) > best_size:
                                    best_set = S.copy()
                                    best_size = len(S)

    # Return the best valid set found
    if best_set:
        is_valid, _ = is_valid_and_count(best_set)
        if is_valid:
            return list(best_set)
    
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
