j_local = l_idx // block_size  # j_local = index // 3
k_local = l_idx % block_size   # k_local = index % 3
g_idx = j_local * p + (base_k + k_local)
