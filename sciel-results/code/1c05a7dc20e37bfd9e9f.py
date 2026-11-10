   div_sum = np.zeros(ten_K + 1, dtype=np.float64)
   for k in range(1, K + 1):
       if mu[k] != 0:
           div_sum[k::k] += mu[k]
   