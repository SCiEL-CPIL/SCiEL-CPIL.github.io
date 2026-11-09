   count = 0
   for _, y1, y2, y3 in neighbors[x]:
       if S[y1] and S[y2] and S[y3]:
           count += 1
   