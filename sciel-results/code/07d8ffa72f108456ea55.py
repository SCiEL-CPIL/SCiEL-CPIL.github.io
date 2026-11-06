        def solve():
            # Returns True if 2-colorable (SAT)
            
            def bt(state):
                # 1. Propagate
                new_state = state[:]
                changed = True
                while changed:
                    changed = False
                    for mask in edge_masks:
                        c0 = 0
                        c1 = 0
                        unassigned = -1
                        for v in range(n):
                            if mask & (1 << v):
                                if new_state[v] == 1: c0 += 1
                                elif new_state[v] == 2: c1 += 1
                                else: unassigned = v
                        if c0 == 5 or c1 == 5:
                            return False # Conflict
                        if c0 == 4 and unassigned != -1:
                            if new_state[unassigned] == 0:
                                new_state[unassigned] = 2
                                changed = True
                            elif new_state[unassigned] == 1:
                                return False # Conflict
                        if c1 == 4 and unassigned != -1:
                            if new_state[unassigned] == 0:
                                new_state[unassigned] = 1
                                changed = True
                            elif new_state[unassigned] == 2:
                                return False # Conflict
                
                # 2. Branch
                v = -1
                for i in range(n):
                    if new_state[i] == 0:
                        v = i
                        break
                        
                if v == -1:
                    return True # All assigned, no conflict -> SAT
                
                # Try 0
                new_state[v] = 1
                if bt(new_state):
                    return True
                
                # Try 1
                new_state[v] = 2
                if bt(new_state):
                    return True
                
                return False

            return bt([0]*n)
