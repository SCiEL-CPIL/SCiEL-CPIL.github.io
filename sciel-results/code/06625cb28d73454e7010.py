   def _distinct_subset_products(nums):
       k = len(nums)
       if k == 0:
           return True
       vecs = [_VECTORS[a] for a in nums]
       seen = {tuple([0] * _NP)}
       cur = [tuple([0] * _NP)]
       for v in vecs:
           new = []
           for s in cur:
               t = tuple(s[i] + v[i] for i in range(_NP))
               if t in seen:
                   return False
               seen.add(t)
               new.append(t)
           cur = new
       return True
   