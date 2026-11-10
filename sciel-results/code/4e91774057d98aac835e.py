    def check_feasibility(mod_list):
        if not mod_list:
            return True, {}
        
        sorted_mods = sorted(mod_list)
        residues = {}
        nodes_visited = [0]
        limit = 10000
        
        def backtrack(idx):
            if nodes_visited[0] > limit:
                return False
            if idx == len(sorted_mods):
                return True
            
            m = sorted_mods[idx]
            for r in range(m):
                nodes_visited[0] += 1
                if nodes_visited[0] > limit:
                    return False
                ok = True
                for i in range(idx):
                    m_i = sorted_mods[i]
                    r_i = residues[m_i]
                    if (r - r_i) % gcd_table[m][m_i] == 0:
                        ok = False
                        break
                if ok:
                    residues[m] = r
                    if backtrack(idx + 1):
                        return True
                    del residues[m]
            return False
            
        return backtrack(0), residues
