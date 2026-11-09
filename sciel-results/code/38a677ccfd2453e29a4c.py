import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 512
    MAXC = 48
    cache = {}

    def build_dp(S):
        """
        Builds DP bitsets for subset sums of S.
        dp[k] is a bitset where bit s is 1 if s is the sum of exactly k elements from S.
        """
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1  # Sum 0 with 0 elements
        for a in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def get_dps(S):
        """
        Caches the full DP and the DPs for S without each element x.
        """
        key = tuple(sorted(S))
        r = cache.get(key)
        if r is not None:
            return r
        
        dp_full = build_dp(S)
        maxfull = sum(S)
        dp_without = {}
        n = len(S)
        
        for i in range(n):
            x = S[i]
            sub = S[:i] + S[i + 1:]
            dp_sub = build_dp(sub)
            max_sub = maxfull - x
            dp_without[x] = (dp_sub, max_sub)
            
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        """
        Checks if x can be added to S without violating the condition.
        """
        n = len(S)
        
        # Check 1: x is mean of subset of S
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False

        # Check 2: a in S is mean of subset involving x
        for a in S:
            dpo, maxo = dp_without[a]
            for j in range(1, n):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dpo[j] >> t) & 1:
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

    def count_violations(S):
        """
        Counts the number of violation pairs in S.
        Returns 0 if valid, >0 otherwise.
        """
        n = len(S)
        if n < 3:
            return 0
        Sset = set(S)
        count = 0
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
                    count += 1
                    # Early exit: if we find one violation, return a large number
                    # to indicate invalidity quickly
                    return count + 1000
        return count

    def greedy_initial_solution(order):
        """
        Greedily construct a valid set from the given order.
        """
        cur = []
        dps = None
        for x in order:
            if len(cur) >= MAXC:
                break
            if time.time() > deadline - 10:
                break
            if dps is None:
                dps = get_dps(cur)
            if test_add(cur, x, dps[0], dps[1], dps[2]):
                cur.append(x)
                dps = None
        return cur

    # Generate initial candidate ordering using constraint counting
    # Elements that are more likely to be means of other subsets should go first
    # Heuristic: prefer numbers with fewer divisors and that are not near the center
    all_nums = list(range(1, N + 1))
    
    # Compute a "constraint score" for each number
    def constraint_score(x):
        """
        Lower score = fewer constraints = should be placed earlier.
        We want to place "flexible" numbers first.
        """
        score = 0
        # Numbers with many divisors are more likely to be means
        div_count = sum(1 for d in range(1, x + 1) if x % d == 0)
        score += div_count * 2
        
        # Numbers near the middle of the range are more likely to be means
        mid = N / 2
        score += abs(x - mid) / 10
        
        # Smaller numbers have fewer possible subset sums that could equal them
        score += x / 100
        
        return score
    
    # Sort by constraint score (ascending - flexible first)
    sorted_by_constraint = sorted(all_nums, key=lambda x: constraint_score(x))
    
    # Create multiple orderings
    orders = [sorted_by_constraint]
    
    # Add some random shuffles for diversity
    for _ in range(3):
        mixed = all_nums[:]
        random.shuffle(mixed)
        orders.append(mixed)
    
    # Phase 1: Generate initial solutions using greedy
    best = []
    best_sz = 0
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        cur = greedy_initial_solution(order)
        if len(cur) > best_sz and verify(cur):
            best_sz = len(cur)
            best = cur[:]
    
    # Phase 2: Simulated Annealing
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        random.shuffle(unused)
        
        # Initial temperature
        T = 10.0
        cooling_rate = 0.95
        min_T = 0.01
        
        best_sa = S[:]
        best_sa_sz = len(S)
        cur_violations = 0 if verify(S) else count_violations(S)
        
        no_improve = 0
        step = 0
        
        while T > min_T and time.time() < deadline - 2:
            step += 1
            if time.time() > deadline - 2:
                break
            
            # Choose a random move
            move_type = random.random()
            
            newS = S[:]
            newSset = set(S)
            
            if len(S) < MAXC and move_type < 0.3:
                # Add a random unused element
                if unused:
                    u = random.choice(unused)
                    newS.append(u)
                    newSset.add(u)
            elif move_type < 0.6:
                # Remove a random element
                if S:
                    i = random.randint(0, len(S) - 1)
                    s = S[i]
                    newS.pop(i)
                    newSset.remove(s)
            elif move_type < 0.85:
                # Swap: remove one, add one
                if S and unused:
                    i = random.randint(0, len(S) - 1)
                    s = S[i]
                    u = random.choice(unused)
                    newS[i] = u
                    newSset.remove(s)
                    newSset.add(u)
            else:
                # 2-swap: remove two, add two
                if len(S) >= 2 and len(unused) >= 2:
                    i1 = random.randint(0, len(S) - 1)
                    i2 = random.randint(0, len(S) - 1)
                    if i1 == i2:
                        i2 = (i2 + 1) % len(S)
                    s1, s2 = S[i1], S[i2]
                    u1, u2 = random.sample(unused, 2)
                    newS[i1] = u1
                    newS[i2] = u2
                    newSset.remove(s1)
                    newSset.remove(s2)
                    newSset.add(u1)
                    newSset.add(u2)
            
            # Check if the move is valid
            new_violations = 0
            if len(newS) >= 3:
                # Quick check: if the size changed, we need to recompute
                # For efficiency, only do full verification if the move seems promising
                if len(newS) != len(S) or True:  # Always verify for now
                    new_violations = count_violations(newS)
            else:
                new_violations = 0
            
            # Accept/reject based on simulated annealing
            delta = new_violations - cur_violations
            
            if delta <= 0:
                # Accept improvement or equal
                S = newS
                Sset = newSset
                cur_violations = new_violations
                no_improve = 0
                
                if cur_violations == 0 and len(S) > best_sa_sz:
                    best_sa = S[:]
                    best_sa_sz = len(S)
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < pow(2.718281828, -delta / T):
                    S = newS
                    Sset = newSset
                    cur_violations = new_violations
                    no_improve += 1
                else:
                    no_improve += 1
            
            # Update unused set
            unused = [x for x in range(1, N + 1) if x not in Sset]
            
            # Cool down
            T *= cooling_rate
            
            # Early stop if stuck
            if no_improve > 200:
                break
        
        # If SA found a better solution, use it
        if verify(best_sa) and len(best_sa) > best_sz:
            best_sz = len(best_sa)
            best = best_sa[:]

    # Final safety check
    if not verify(best):
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
