   def gen_sums(w):
       sums = [(0, 0)]
       for wi in w:
           new_sums = []
           for s, m in sums:
               new_sums.append((s + wi, m * 3 + 1))
               new_sums.append((s - wi, m * 3 + 2))
               new_sums.append((s, m * 3 + 0))
           sums = new_sums
       return sums
   