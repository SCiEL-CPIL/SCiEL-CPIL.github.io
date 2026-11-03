local_i_prime = l // 2  # 0 or 1
local_k_prime = l % 2   # 0 or 1
global_i = 1 + local_i_prime
global_k = k_start + local_k_prime
W[global_k * n + global_i][col] += strassen_W[t][l]
