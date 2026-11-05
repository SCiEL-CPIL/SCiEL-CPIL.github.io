"""Rewritten program: greedy Egyptian fraction with denominators > 4."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return a list of distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1, minimizing the count.
    
    Uses a greedy/Sylvester-style approach: at each step, choose the
    smallest integer d >= 5 such that 1/d <= remaining and d doesn't
    repeat, but we need to be careful to ensure we can complete the sum.
    
    Actually, for minimizing count, we want as few terms as possible.
    The minimum number of terms for an Egyptian fraction of 1 with
    denominators >= 5 is known to be small. Let's think:
    
    With denominators >= 5, the maximum reciprocal is 1/5.
    To sum to 1, we need at least 5 terms (5 * 1/5 = 1, but that requires
    all denominators = 5, which aren't distinct).
    
    With distinct denominators >= 5, the sum of reciprocals:
    1/5 + 1/6 + 1/7 + 1/8 + 1/9 + ... 
    We need to find the minimal k such that there exist distinct d1 < d2 < ... < dk
    all >= 5 with sum(1/di) = 1.
    
    Let's try small k:
    k=5: max possible sum with 5 distinct denominators >= 5 is 1/5+1/6+1/7+1/8+1/9
    = (2520*... let's compute: 1/5=0.2, 1/6≈0.1667, 1/7≈0.1429, 1/8=0.125, 1/9≈0.1111
    Sum ≈ 0.7457 < 1. So k=5 is impossible.
    
    k=6: 1/5+1/6+1/7+1/8+1/9+1/10 = 0.2+0.1667+0.1429+0.125+0.1111+0.1 = 0.8457 < 1.
    Still < 1.
    
    k=7: add 1/11 ≈ 0.0909, sum ≈ 0.9366 < 1.
    k=8: add 1/12 ≈ 0.0833, sum ≈ 1.0199 > 1. So k=8 is possible.
    
    So the minimum is likely 8 terms. Let's find an explicit 8-term representation.
    
    We need distinct d1..d8 >= 5 with sum 1/di = 1.
    
    Let me try to construct one. Start with 1/5, remaining = 4/5.
    Then 1/6, remaining = 4/5 - 1/6 = 24/30 - 5/30 = 19/30.
    Then 1/7, remaining = 19/30 - 1/7 = (133-30)/210 = 103/210.
    Then 1/8, remaining = 103/210 - 1/8 = (824-210)/1680 = 614/1680 = 307/840.
    Then 1/9, remaining = 307/840 - 1/9 = (2763-840)/7560 = 1923/7560 = 641/2520.
    Then 1/10, remaining = 641/2520 - 1/10 = (6410-2520)/25200 = 3890/25200 = 389/2520.
    Then 1/11, remaining = 389/2520 - 1/11 = (4279-2520)/27720 = 1759/27720.
    Then we need 1/d8 = 1759/27720, so d8 = 27720/1759 ≈ 15.76. Not integer.
    
    This greedy approach doesn't work directly. Let me try a different set.
    
    Known result: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... no, that's too many.
    
    Let me try: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/14 + 1/15 + 1/20 + 1/30 + 1/42 + ... no.
    
    Let me use a known identity. Actually, let me just hardcode a known 8-term Egyptian fraction of 1 with denominators >= 5.
    
    One known representation: 1 = 1/2 + 1/3 + 1/7 + 1/42, but 2 and 3 are < 5.
    
    Let me try to find one computationally with a small search.
    """
    from fractions import Fraction
    
    # Try to find a minimal representation with a targeted search
    # We know k >= 8. Let's try k=8 first.
    
    def find_egyptian(target, min_den, k, used=None):
        """Find k distinct denominators >= min_den summing to target."""
        if used is None:
            used = set()
        if k == 0:
            return [] if target == 0 else None
        if k == 1:
            d = 1 / target
            if d == int(d) and int(d) >= min_den and int(d) not in used:
                return [int(d)]
            return None
        # The smallest possible denominator is min_den
        # The largest possible denominator: if we pick d, remaining target' = target - 1/d
        # For the remaining k-1 terms with min denominator d+1, max sum is (k-1)/(d+1)
        # So we need target - 1/d <= (k-1)/(d+1)
        # Also 1/d <= target
        # And d >= min_den
        max_d = int(1 / target) + 1
        # Also need enough remaining sum
        # (k-1)/(d+1) >= target - 1/d
        for d in range(min_den, min(max_d, 10000000) + 1):
            if d in used:
                continue
            if Fraction(1, d) > target:
                break
            remaining = target - Fraction(1, d)
            if remaining < 0:
                continue
            # Check if remaining can be achieved with k-1 terms >= d+1
            # Max possible: (k-1)/(d+1)
            if remaining > Fraction(k-1, d+1):
                continue
            # Min possible: sum of 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1)
            min_sum = sum(Fraction(1, d+i) for i in range(1, k))
            if remaining < min_sum:
                # Even the smallest possible sum is too large, so larger d will make it worse
                # Actually no, larger d means smaller reciprocals, so min_sum decreases
                # But we're iterating d upward, so let's just continue
                pass
            result = find_egyptian(remaining, d + 1, k - 1, used | {d})
            if result is not None:
                return [d] + result
        return None
    
    # Try k=8
    for k in range(8, 13):
        result = find_egyptian(Fraction(1, 1), 5, k)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            assert total == 1, f"Sum is {total}, not 1"
            assert all(d >= 5 and d <= 10000000 for d in result)
            assert len(set(result)) == len(result)
            return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
