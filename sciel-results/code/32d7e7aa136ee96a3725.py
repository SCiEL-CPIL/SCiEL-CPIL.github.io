<<<<<<< SEARCH
        type_to_grid = [support_list[i][0]*4 + support_list[i][1] for i in range(16)]
        grid_to_type = [0]*16
        for i in range(16):
            grid_to_type[type_to_grid[i]] = i
=======
        type_to_grid = [int(support_list[i][0])*4 + int(support_list[i][1]) for i in range(16)]
        grid_to_type = [0]*16
        for i in range(16):
            g_idx = int(type_to_grid[i])
            if 0 <= g_idx < 16:
                grid_to_type[g_idx] = i
            else:
                grid_to_type[i] = i # Fallback
>>>>>>> REPLACE
