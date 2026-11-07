   part_a = [0, 1, 2, 3, 4, 5, 6, 7]
   part_b = [8, 9, 10, 11, 12, 13, 14, 15]
   edge_list = []
   for a in part_a:
       for b in part_b:
           edge_list.append([int(a), int(b)])
   return edge_list
   