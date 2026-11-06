            colors = [0] * (target_n * (target_n - 1) // 2)
            
            def get_idx(i, j):
                return i * (target_n - 1) - i * (i + 1) // 2 + (j - i - 1)

            for v in range(1, target_n):
                c_list = history[v-1]
                for u in range(v):
                    c = c_list[u]
                    idx = get_idx(u, v)
                    colors[idx] = c
