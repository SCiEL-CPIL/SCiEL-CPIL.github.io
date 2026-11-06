def construct():
    # We want a 32-element set S in 1..512 that maximizes the number of distinct d > 0
    # such that there exists a with a, a+d, a+2d in S.
    
    # A contiguous block [1..32] gives d in 1..15 (since 1, 1+d, 1+2d <= 32 => 2d <= 31 => d <= 15.5)
    # So score = 15.
    
    # If we use [1..31] U {47}:
    # Core [1..31] gives d in 1..15.
    # Outlier 47 with a in [1..31]: need a, (a+47)/2, 47 in S.
    # (a+47)/2 must be in [1..31] and integer.
    # a+47 even => a odd.
    # (a+47)/2 <= 31 => a+47 <= 62 => a <= 15.
    # a >= 1.
    # So a in {1, 3, 5, 7, 9, 11, 13, 15}.
    # For each such a, d = (47-a)/2.
    # a=1 => d=23
    # a=3 => d=22
    # a=5 => d=21
    # a=7 => d=20
    # a=9 => d=19
    # a=11 => d=18
    # a=13 => d=17
    # a=15 => d=16
    # These are all new (16..23).
    # Total distinct d: 1..15 (from core) U 16..23 (from outlier) = 23.
    
    # Let's verify if we can do better.
    # What if we use two outliers?
    # S = [1..30] U {47, 63}?
    # Core [1..30] gives d in 1..14.
    # Outlier 47 with a in [1..30]:
    # a odd, a <= 15.
    # a=1 => d=23
    # a=3 => d=22
    # ...
    # a=15 => d=16
    # So 16..23.
    # Outlier 63 with a in [1..30]:
    # a+63 even => a odd.
    # (a+63)/2 <= 30 => a+63 <= 60 => a <= -3. Impossible.
    # So 63 doesn't help with a in [1..30].
    # What about a in {47}?
    # 47, (47+63)/2=55, 63. 55 not in S.
    # So 63 is useless.
    
    # What if S = [1..29] U {47, 63}?
    # Core [1..29] gives d in 1..14.
    # Outlier 47 with a in [1..29]:
    # a odd, a <= 15.
    # 16..23.
    # Outlier 63 with a in [1..29]:
    # a odd, a <= -3. Impossible.
    # Outlier 63 with a in {47}:
    # 47, 55, 63. 55 not in S.
    
    # What if S = [1..28] U {47, 63, 79}?
    # Core [1..28] gives d in 1..13.
    # Outlier 47 with a in [1..28]:
    # a odd, a <= 15.
    # 16..23.
    # Outlier 63 with a in [1..28]:
    # a odd, a <= -3. Impossible.
    # Outlier 63 with a in {47}:
    # 47, 55, 63. 55 not in S.
    # Outlier 79 with a in [1..28]:
    # a+79 even => a odd.
    # (a+79)/2 <= 28 => a+79 <= 56 => a <= -23. Impossible.
    
    # It seems placing outliers far away only helps if the midpoint falls in the core.
    # For outlier X and core [1..N], we need a in [1..N] such that (a+X)/2 in [1..N].
    # This implies a <= 2N - X.
    # If X is large, 2N - X becomes small or negative.
    # For N=31, X=47, 2N-X = 62-47 = 15. So a in 1..15 (odd).
    # For N=30, X=47, 2N-X = 60-47 = 13. So a in 1..13 (odd).
    #   d values: (47-1)/2=23, (47-3)/2=22, ..., (47-13)/2=17.
    #   So 17..23. (7 values).
    #   Core gives 1..14.
    #   Total 14 + 7 = 21. Worse than 23.
    
    # So [1..31] U {47} seems better.
    # Can we get more from the outlier?
    # If we use X=49?
    # S = [1..31] U {49}.
    # a+49 even => a odd.
    # (a+49)/2 <= 31 => a+49 <= 62 => a <= 13.
    # a in {1, 3, 5, 7, 9, 11, 13}.
    # d = (49-a)/2.
    # a=1 => d=24
    # a=3 => d=23
    # ...
    # a=13 => d=18
    # New d: 18..24. (7 values).
    # Core 1..15.
    # Total 15 + 7 = 22. Worse.
    
    # If we use X=45?
    # S = [1..31] U {45}.
    # a+45 even => a odd.
    # (a+45)/2 <= 31 => a+45 <= 62 => a <= 17.
    # a in {1, 3, 5, 7, 9, 11, 13, 15, 17}.
    # d = (45-a)/2.
    # a=1 => d=22
    # ...
    # a=17 => d=14.
    # New d: 14..22.
    # But 14 is already in core.
    # So new: 15..22. (8 values).
    # Core 1..15.
    # Total 15 + 8 = 23. Same as X=47.
    
    # Let's check X=43.
    # a+43 even => a odd.
    # (a+43)/2 <= 31 => a+43 <= 62 => a <= 19.
    # a in {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}.
    # d = (43-a)/2.
    # a=1 => d=21
    # ...
    # a=19 => d=12.
    # New d: 12..21.
    # Core 1..15.
    # Overlap 12..15.
    # New: 16..21. (6 values).
    # Total 15 + 6 = 21. Worse.
    
    # So X=45 and X=47 give 23.
    # Are there any other sets that give more?
    # What if we don't use a contiguous core?
    # But contiguous core maximizes the number of small d's.
    # The number of d's from a set of size N is at most floor((N-1)/2) if it's an interval.
    # For N=31, that's 15.
    # To get more than 15 small d's, we would need a non-interval set that is "denser" in some sense,
    # but any set of 31 integers in [1..M] has at most 15 distinct d's if M is small?
    # No, e.g. {1, 2, 4, 5, 7, 8, ...} might have more?
    # Actually, the maximum number of distinct differences in a set of size N is N(N-1)/2,
    # but we are looking for 3-term APs.
    # The number of d's such that a, a+d, a+2d in S is related to the structure.
    
    # Let's stick with the interval + outlier strategy.
    # Is there a way to get more than 23?
    # What if we use two outliers that both contribute?
    # S = [1..30] U {47, 63}?
    # We saw 63 doesn't help.
    # What if S = [1..29] U {47, 65}?
    # Core [1..29] gives 1..14.
    # 47 with a in [1..29]: a odd, a <= 13.
    # d: 17..23. (7 values).
    # 65 with a in [1..29]: a odd, a <= -1. Impossible.
    # 65 with a in {47}: 47, 56, 65. 56 not in S.
    
    # What if the outliers are closer?
    # S = [1..30] U {35, 40}?
    # Core [1..30] gives 1..14.
    # 35 with a in [1..30]:
    # a+35 even => a odd.
    # (a+35)/2 <= 30 => a+35 <= 60 => a <= 25.
    # a in {1, 3, ..., 25}.
    # d = (35-a)/2.
    # a=1 => d=17
    # ...
    # a=25 => d=5.
    # New d: 5..17.
    # Overlap 5..14.
    # New: 15, 16, 17. (3 values).
    # 40 with a in [1..30]:
    # a+40 even => a even.
    # (a+40)/2 <= 30 => a+40 <= 60 => a <= 20.
    # a in {2, 4, ..., 20}.
    # d = (40-a)/2.
    # a=2 => d=19
    # ...
    # a=20 => d=10.
    # New d: 10..19.
    # Overlap 10..14.
    # New: 15, 16, 17, 18, 19.
    # But 15, 16, 17 already added by 35.
    # So new from 40: 18, 19.
    # Total new: 15, 16, 17, 18, 19. (5 values).
    # Total score: 14 + 5 = 19. Worse.
    
    # It seems the single outlier strategy is best.
    # Max score found: 23.
    
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 47]
