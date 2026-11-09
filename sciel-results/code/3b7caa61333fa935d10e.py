import time
import math

def construct():
    start_time = time.time()
    limit_time = 55.0  # Safety margin for 60s limit

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def dfs(N, D, min_denom, terms_left, current):
        # N/D is the remaining sum
        if terms_left == 0:
            return current[:] if N == 0 else None
        
        if terms_left == 1:
            # Need 1/d = N/D => d = D/N
            if N == 0:
                return None
            if D % N == 0:
                d = D // N
                if d >= min_denom and d <= 10000000:
                    return current + [d]
            return None

        # Pruning: Max sum of remaining terms
        # Sum approx (terms_left) / min_denom
        # If terms_left / min_denom < N/D, it's impossible
        # More precise: sum(1/i for i in range(min_denom, min_denom + terms_left))
        # For small terms_left, we can compute exactly or use a tight bound.
        # Let's use a simple bound first: terms_left / min_denom < N/D
        if terms_left * D < N * min_denom:
            return None

        # Lower bound for d: 1/d <= N/D => d >= D/N
        min_d = max(min_denom, (D + N - 1) // N)
        
        # Upper bound for d:
        # We need to leave enough sum for terms_left - 1 terms.
        # N/D - 1/d <= (terms_left - 1) / (d + 1)  (very loose, but helps)
        # Actually, the max sum of remaining terms starting from d+1 is approx (terms_left-1)/(d+1).
        # So N/D - 1/d <= (terms_left-1)/(d+1)
        # N/D <= 1/d + (terms_left-1)/(d+1)
        # For large d, RHS ~ terms_left/d. So d <= terms_left * D / N.
        # Let's use a slightly tighter bound.
        # d <= (terms_left * D) // N + 1
        max_d = min(10000000, (terms_left * D) // N + 2)
        
        # Iterate d
        for d in range(min_d, max_d + 1):
            if time.time() - start_time > limit_time:
                return "TIMEOUT"
            
            # New remaining fraction: N/D - 1/d = (Nd - D) / (Dd)
            new_N = N * d - D
            if new_N < 0:
                # 1/d > N/D, so we skipped valid d's? 
                # No, min_d ensures 1/d <= N/D.
                # If new_N < 0, it means 1/d > N/D, which contradicts d >= D/N.
                # Wait, if d = D/N, new_N = 0.
                # If d > D/N, new_N > 0.
                # So new_N should be >= 0.
                continue
            
            if new_N == 0:
                # We have used 1 term to consume the sum, but we have terms_left - 1 > 0 terms left.
                # This is only valid if terms_left == 1, which is handled above.
                # So if terms_left > 1 and new_N == 0, we have extra terms that must sum to 0, impossible.
                continue

            new_D = D * d
            
            # Reduce fraction
            g = gcd(new_N, new_D)
            if g > 1:
                new_N //= g
                new_D //= g

            result = dfs(new_N, new_D, d + 1, terms_left - 1, current + [d])
            if result is not None:
                return result
        
        return None

    # Try k from 8 to 256
    for k in range(8, 257):
        # Initial state: N=1, D=1, min_denom=5, terms_left=k
        res = dfs(1, 1, 5, k, [])
        if res is not None and res != "TIMEOUT":
            # Verify
            total = 0
            for d in res:
                total += 1.0 / d
            # Check exactness using Fraction for verification
            from fractions import Fraction
            exact_sum = sum(Fraction(1, d) for d in res)
            if exact_sum == 1 and all(5 <= d <= 10000000 for d in res) and len(set(res)) == len(res):
                return res
        elif res == "TIMEOUT":
            break

    # Fallback: Return a known valid solution if search fails (should not happen for k=8)
    # A known solution for k=11?
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This is not 1.
    
    # Let's try to find a hardcoded solution for k=8 or 9 if search is too slow.
    # But the search should be fast for k=8.
    
    raise RuntimeError("No solution found")
