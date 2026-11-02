    def greedy_reduce(order):
        covered_count = [0] * total_words
        code = []
        
        # Greedy phase
        for w_idx in order:
            # Check if this word adds any new coverage (count goes from 0 to 1)
            adds_new = False
            for u in balls[w_idx]:
                if covered_count[u] == 0:
                    adds_new = True
                    break
            if adds_new:
                code.append(w_idx)
                for u in balls[w_idx]:
                    covered_count[u] += 1
        
        # Reduction phase
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                w_idx = code[i]
                # Check if redundant: all u in balls[w_idx] have count > 1
                is_redundant = True
                for u in balls[w_idx]:
                    if covered_count[u] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove w_idx
                    code.pop(i)
                    for u in balls[w_idx]:
                        covered_count[u] -= 1
                    changed = True
        
        return code
