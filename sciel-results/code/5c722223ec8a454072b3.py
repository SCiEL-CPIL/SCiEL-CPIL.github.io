   def can_add(S, x):
       S_list = list(S)
       # 1. x does not divide any triple sum in S
       for t in combinations(S_list, 3):
           if sum(t) % x == 0:
               return False
       
       # 2. For each a in S, a does not divide (x + b + c) for any b, c in S \ {a}
       for a in S_list:
           # b, c from S \ {a}
           # We need to check if there exist b, c such that (x + b + c) % a == 0
           # This is equivalent to (b + c) % a == (-x) % a
           target = (-x) % a
           # Find if any pair b, c in S \ {a} sums to target mod a
           # We can use a set of residues?
           # Or just loop b, c.
           # S \ {a} has size ~47. Pairs ~1100.
           # 47 * 1100 = 51k ops per a. 47 * 51k = 2.4M ops.
           # This is too slow for 480k checks.
           
           # Optimization:
           # Precompute pair sums mod a for S \ {a}?
           # No, a changes.
           
           # Alternative:
           # For fixed a and x, we need to know if there are b, c in S\a such that b+c == k*a - x.
           # Since b,c < 192, b+c < 384.
           # So k is small.
           # We can precompute all pair sums in S.
           # Let pair_sums be a set of (sum, b, c)?
           # If we have a set of all pair sums in S, we can check if (target_sum) is in pair_sums.
           # But we must ensure b, c != a and b != c.
           # If we just check if the sum exists, we might pick b=a.
           # We can store counts or just verify after finding a candidate sum.
           
           # Let's precompute `all_pair_sums` for S.
           # `all_pair_sums` is a dictionary: sum -> list of (b,c)
           # Then for each a, we check if there is a pair sum s such that (s + x) % a == 0.
           # And the pair (b,c) does not contain a.
           
           # This reduces the check to iterating over possible sums?
           # Number of distinct pair sums is small (max 384).
           # So for each a, we iterate over possible sums s (1..384)?
           # Or just iterate over the keys of the pair sum dict.
           # Number of keys is at most 384.
           # So for each a, we do ~384 checks.
           # 48 * 384 = 18k ops.
           # This is much faster than 2.4M.
           
           # So, maintain `pair_sums_dict` for S.
           # Update it when S changes.
   