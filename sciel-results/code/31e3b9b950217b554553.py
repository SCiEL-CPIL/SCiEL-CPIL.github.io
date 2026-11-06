     total_rank = 0
     for i in range(3):
         for j in range(3):
             for k in range(3):
                 ni = block_sizes[i]
                 mj = block_sizes[j]
                 pk = block_sizes[k]
                 if ni == 2 and mj == 2 and pk == 2:
                     total_rank += 7
                 else:
                     total_rank += ni * mj * pk
     