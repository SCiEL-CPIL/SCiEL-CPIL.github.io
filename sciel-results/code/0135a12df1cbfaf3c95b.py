            found_swap = False
            # Try adjacent swaps
            for i in range(len(current_net) - 1):
                candidate = current_net[:]
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                if _is_valid(candidate):
                    current_net = candidate
                    best_net = candidate # Count same, but structure changed
                    found_swap = True
                    break
            if not found_swap:
                # Try random swaps
                import random
                for _ in range(50):
                    if len(current_net) < 2: break
                    i = random.randint(0, len(current_net) - 1)
                    j = random.randint(0, len(current_net) - 1)
                    if i == j: continue
                    candidate = current_net[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if _is_valid(candidate):
                        current_net = candidate
                        best_net = candidate
                        found_swap = True
                        break
            if not found_swap:
                break
