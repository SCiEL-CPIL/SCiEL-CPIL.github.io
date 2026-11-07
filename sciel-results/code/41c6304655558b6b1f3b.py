"""Program for the suite task: construct a large set in 1..128 with no triple a,b,c
such that a+b, a+c, b+c are all also in the set.
"""

def construct():
    # Use the upper half: 65..128.
    # For any distinct a, b, c in [65, 128], the pair sums a+b, a+c, b+c are in [131, 254].
    # The set S = [65, 128]. The intersection of the sum range with S is [131, 128] which is empty.
    # Wait, 131 > 128. So no pair sum is in S.
    # Therefore, it is impossible for a+b, a+c, b+c to be in S.
    # This set is valid.
    # Size: 128 - 65 + 1 = 64.
    
    # Can we do better?
    # Try starting at 64. S = [64, 128].
    # Pair sums in [128, 254]. Intersection with S: {128}.
    # We need three distinct sums in S. We can have at most one sum in S.
    # So no triple can satisfy the condition that all three sums are in S.
    # So S = [64, 128] is valid. Size: 128 - 64 + 1 = 65.
    
    # Try starting at 63. S = [63, 128].
    # Pair sums in [126, 254]. Intersection with S: {126, 127, 128}.
    # We need three distinct sums from {126, 127, 128} that are a+b, a+c, b+c.
    # Let's check if there exists a, b, c in [63, 128] such that {a+b, a+c, b+c} = {126, 127, 128}.
    # Sum of all three sums is 2(a+b+c) = 126+127+128 = 381.
    # 2(a+b+c) = 381 => a+b+c = 190.5, which is not an integer.
    # So no such triple exists.
    # Are there other combinations?
    # The sums don't have to be exactly {126, 127, 128}, they just have to be in S.
    # But the only elements of S that are >= 126 are 126, 127, 128.
    # So any valid triple must have its sums exactly be {126, 127, 128}.
    # Since that's impossible, S = [63, 128] is valid. Size: 128 - 63 + 1 = 66.
    
    # Try starting at 62. S = [62, 128].
    # Pair sums in [124, 254]. Intersection with S: {124, 125, 126, 127, 128}.
    # We need three distinct sums from this set of 5.
    # Let's check if a triple exists.
    # We need a+b, a+c, b+c in {124, 125, 126, 127, 128}.
    # Let's try to find a, b, c.
    # If we pick small a, b, c from the set.
    # a=62, b=63. a+b=125.
    # Need c such that 62+c and 63+c are in {124, 126, 127, 128} (excluding 125).
    # 62+c >= 124 => c >= 62.
    # 63+c >= 124 => c >= 61.
    # Let c=64.
    # 62+64=126. 63+64=127.
    # Sums: 125, 126, 127. All in S.
    # So (62, 63, 64) is a violating triple.
    # Thus [62, 128] is not valid.
    
    # So the contiguous block [63, 128] of size 66 is valid.
    # Can we add 62 to it by removing some elements?
    # The violation was (62, 63, 64).
    # If we remove 64, can we keep 62?
    # New set S' = {62, 63} U {65..128}.
    # Check triples involving 62, 63.
    # a=62, b=63. a+b=125.
    # Need c in S' such that 62+c in S' and 63+c in S'.
    # c >= 65.
    # 62+65=127. 63+65=128.
    # 127 in S'? Yes. 128 in S'? Yes.
    # So (62, 63, 65) is a violation.
    # If we remove 65 too?
    # S'' = {62, 63} U {66..128}.
    # a=62, b=63. a+b=125.
    # c=66. 62+66=128. 63+66=129 (not in S'').
    # c=67. 62+67=129 (not in S'').
    # So for a=62, b=63, no c works?
    # Let's check other pairs in {62, 63}.
    # What if c=62? Distinct members required.
    # So we need to check if any c in {66..128} works.
    # 62+c in [128, 190]. Intersection with S'': {128}. So 62+c must be 128 => c=66.
    # 63+c in [129, 191]. Intersection with S'': none.
    # So (62, 63, c) never works.
    # But we must check all triples.
    # What about triples not involving 62 or 63?
    # Those are in {66..128}.
    # Sums in [132, 254]. Intersection with S'': none (max 128).
    # So no violation there.
    # What about triples involving one of 62,63 and two from {66..128}?
    # a=62, b=c? No, distinct.
    # a=62, b in {66..128}, c in {66..128}.
    # b+c >= 66+67=133. Not in S''.
    # So this is safe.
    # What about a=63, b in {66..128}, c in {66..128}.
    # b+c >= 133. Not in S''.
    # Safe.
    # So S'' = {62, 63, 66, 67, ..., 128} is valid.
    # Size: 2 + (128-66+1) = 2 + 63 = 65.
    # This is smaller than 66.
    
    # Can we get 67?
    # Maybe start at 62 and remove more?
    # Or use a different structure?
    # The upper bound for sum-free sets is roughly N/2.
    # 128/2 = 64.
    # We found 66.
    # Is 67 possible?
    # If we take [62, 128] and fix violations.
    # Violations in [62, 128]:
    # (62,63,64), (62,63,65), (62,64,66)?
    # 62+64=126, 62+66=128, 64+66=130(no).
    # (62,64,66) sums: 126, 128, 130. 130 not in S. So no.
    # (62,63,64) sums: 125, 126, 127. All in S.
    # (62,63,65) sums: 125, 127, 128. All in S.
    # (62,64,65) sums: 126, 127, 129(no).
    # (63,64,65) sums: 127, 128, 129(no).
    # (62,63,66) sums: 125, 128, 129(no).
    # (62,64,67) sums: 126, 130(no)...
    # (62,65,66) sums: 127, 128, 131(no).
    # (63,64,66) sums: 127, 129(no)...
    # (63,65,66) sums: 128, 129(no)...
    # (64,65,66) sums: 129, 130, 131. None in S.
    
    # So the only problematic triples in [62, 128] seem to involve small numbers.
    # We removed 64 and 65 to get size 65.
    # Can we keep 64?
    # If we keep 64, we must remove 63? Or 62?
    # If we remove 62: S = {63..128}. Size 66. (Valid)
    # If we remove 63: S = {62, 64..128}.
    # Check (62, 64, 66)? 126, 128, 130(no).
    # Check (62, 64, 65)? 126, 127, 129(no).
    # Check (62, 65, 66)? 127, 128, 131(no).
    # Check (64, 65, 66)? 129...
    # Check (62, 64, 67)? 126, 130(no).
    # Check (62, 65, 67)? 127, 131(no).
    # Check (64, 65, 67)? 129...
    # Check (62, 66, 67)? 128, 132(no).
    # Check (62, 64, 68)? 126, 132(no).
    # It seems {62, 64..128} might be valid.
    # Size: 1 + (128-64+1) = 1 + 65 = 66.
    # Same as 66.
    
    # Can we add 61?
    # S = {61, 62, 64..128}?
    # (61, 62, 64): 123, 125, 126.
    # 123 in S? Yes. 125 in S? Yes. 126 in S? Yes.
    # Violation.
    # So we can't just add 61.
    
    # Let's stick with 66.
    # Return list(range(63, 129))
    
    return list(range(63, 129))

# EVOLVE-BLOCK-END
