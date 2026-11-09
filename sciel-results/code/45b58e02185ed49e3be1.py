import time
import random

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
            # Update in reverse to avoid using the same element multiple times in one step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= dp[k] << a
        return dp

    def get_dps(S):
        """
        Caches the full DP and the DPs for S without each element x.
        Returns: (dp_full, max_sum_full, {x: (dp_without_x, max_sum_without_x)})
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
            sub = S[:i] + S[i + 1:]
            dp_without[S[i]] = (build_dp(sub), maxfull - S[i])
        
        r = (dp_full, maxfull, dp_without)
        cache[key] = r
        return r

    def test_add(S, x, dp_full, maxfull, dp_without):
        """
        Checks if x can be added to S without violating the mean condition.
        1. x is not the mean of any subset of S (size >= 2).
        2. No element a in S becomes the mean of any subset of S U {x} containing x.
        """
        n = len(S)
        # Check 1: x not the mean of any subset of S (size>=2)
        for k in range(2, n + 1):
            t = k * x
            if t > maxfull:
                break
            if (dp_full[k] >> t) & 1:
                return False
        
        # Check 2: no a in S becomes mean of a subset containing x
        # Subset T of S U {x} \ {a} has size j+1, sum (j+1)*a.
        # T contains x. Let T' = T \ {x}. T' is subset of S \ {a} of size j.
        # Sum(T') = (j+1)*a - x.
        for a in S:
            dpo, maxo = dp_without[a]
            m = n - 1  # Size of S \ {a}
            # j can range from 1 to m (since T size >= 2, and T contains x, T' size >= 1)
            # Max T size is n (since S U {x} has n+1 elements, remove a -> n elements).
            # So j can be up to n-1.
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0 or t > maxo:
                    continue
                if (dpo[j] >> t) & 1:
                    return False
        return True

    def verify(S):
        """
        Verifies the entire set S.
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

    best = []
    best_sz = 0

    # Phase 1: Multi-order Greedy
    # Use random orders and both directions to find good initial sets
    for i in range(40):
        if time.time() > deadline - 15:
            break
        base = list(range(1, N + 1))
        random.shuffle(base)
        
        for rev in [False, True]:
            order = base[::-1] if rev else base
            cur = []
            dps = None
            for x in order:
                if len(cur) >= MAXC:
                    break
                if dps is None:
                    dps = get_dps(cur)
                if test_add(cur, x, dps[0], dps[1], dps[2]):
                    cur.append(x)
                    dps = None
            
            if verify(cur) and len(cur) > best_sz:
                best_sz = len(cur)
                best = cur[:]
        
        if best_sz >= MAXC:
            break

    # Phase 2: Local Search (add + swap)
    if best_sz < MAXC and time.time() < deadline - 5:
        S = best[:]
        Sset = set(S)
        unused = [x for x in range(1, N + 1) if x not in Sset]
        
        random.shuffle(unused)
        
        improved = True
        no_improve_counter = 0
        
        while improved and time.time() < deadline - 2:
            improved = False
            dps = get_dps(S)
            
            # Try to add
            for u in unused[:]:
                if len(S) >= MAXC:
                    break
                if test_add(S, u, dps[0], dps[1], dps[2]):
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
                # Swap an element s in S with u in unused if S \ {s} U {u} is valid
                # We check if u can be added to S \ {s}.
                
                # Shuffle unused to vary order
                random.shuffle(unused)
                
                swapped = False
                # Limit swap search for speed
                # We try all s in S, but limit u checks per s or total
                # Actually, we can just iterate through a subset of unused
                
                # To be safe, iterate through all unused but break if time is tight
                max_swaps_to_try = len(unused)
                if time.time() > deadline - 8:
                    max_swaps_to_try = min(len(unused), 50)

                for u in unused[:max_swaps_to_try]:
                    if time.time() > deadline - 2:
                        break
                    for i, s in enumerate(S):
                        newS = S[:i] + S[i + 1:]
                        # Check if u can be added to newS
                        dps2 = get_dps(newS)
                        if test_add(newS, u, dps2[0], dps2[1], dps2[2]):
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
                    if no_improve_counter > 15:
                        break

        if verify(S) and len(S) > best_sz:
            best_sz = len(S)
            best = S[:]

    # Final safety
    if not verify(best):
        for sz in range(len(best), 0, -1):
            if verify(best[:sz]):
                best = best[:sz]
                break
        else:
            best = []

    return best
