     def is_valid_addition(S, x, sums_set, pair_sums_set, double_S):
         # A vs Old
         if 3*x in sums_set: return False
         
         # B vs Old, C vs Old, D vs Old
         for y in S:
             if (2*x + y) in sums_set: return False
             if (x + 2*y) in sums_set: return False
             
         for p in pair_sums_set:
             if (x + p) in sums_set: return False
             
         # A vs D
         if 2*x in pair_sums_set: return False
         
         # B vs D
         # Check if x+y in pair_sums_set for any y in S
         for y in S:
             if (x + y) in pair_sums_set: return False
             
         # B vs C
         # Check if x+y in double_S for any y in S
         for y in S:
             if (x + y) in double_S: return False
             
         return True
     