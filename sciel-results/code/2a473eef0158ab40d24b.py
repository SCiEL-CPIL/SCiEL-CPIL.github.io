    def can_add(x, S_mask):
        # Find a, b in S such that a+x in S, b+x in S, a+b in S.
        # Let A = {a in S | a+x in S}
        # We need a, b in A such that a+b in S.
        
        # Bitmask of S
        # Bitmask of A:
        # a is in A if bit a is set in S_mask AND bit (a+x) is set in S_mask.
        # Note: a+x <= 128.
        
        # We can construct mask_A.
        # Then we need to check if there are a,b in mask_A such that a+b is in S_mask.
        
        # Iterating over set bits in mask_A is fast if |A| is small.
        # In the beginning S is small, so A is small.
        # Later S is large, A might be large.
        
        # Optimization:
        # If we iterate over a in A, we need to check if (S_mask & (S_mask >> a)) has any bits set in A?
        # No, b must be in A.
        # Check: Is there b in A such that a+b in S?
        # Let T = S_mask >> a. Bits in T correspond to sums a+b.
        # We need to check if any bit set in T is also set in S_mask? No, T *is* the shifted S.
        # We need b such that b in A AND (a+b) in S.
        # This is equivalent to: (A & (S_mask >> a)) != 0 ?
        # Wait. If bit $b$ is set in $A$, and bit $a+b$ is set in $S$, then bit $b$ in $(S >> a)$ is set.
        # So we check if $A \text{ AND } (S \gg a)$ has any set bits.
        # BUT we must ensure $b \neq a$ (distinct a,b).
        # If $b=a$, then $2a \in S$.
        # The problem states "three DISTINCT members a,b,c".
        # So $a \neq b$.
        # If $A \cap (S \gg a)$ contains $a$, that corresponds to $b=a$. We must exclude this.
        
        # So:
        # Candidates = A & (S_mask >> a)
        # If Candidates has any bit set OTHER than bit a, then conflict.
        # If Candidates == (1 << a), it means only b=a works, which is invalid.
        # If Candidates == 0, no conflict for this a.
        
        # So for each a in A:
        #   conflict_mask = A & (S_mask >> a)
        #   if conflict_mask != 0:
        #       if conflict_mask != (1 << a): return False
        #       # else only b=a, which is not a distinct pair, so OK for this a.
        #       # But we must ensure no other b.
        
        # Wait, if conflict_mask has bit a AND bit b (b!=a), then conflict.
        # So:
        #   if (conflict_mask & ~(1 << a)) != 0: return False
        
        # This is $O(|A|)$ bit operations. Very fast.
    