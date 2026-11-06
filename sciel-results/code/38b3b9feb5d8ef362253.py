   for idx in available:
       conflicts = 0
       for i in chosen:
           for j in chosen:
               if i >= j: continue
               if is_sunflower_fast(i, j, idx): ...
   