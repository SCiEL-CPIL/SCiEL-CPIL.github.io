            elif move_type < 0.95:
                # Swap
                i, j = random.sample(range(30), 2)
                new_points[i], new_points[j] = new_points[j], new_points[i]
            else:
                # Random reset
                idx = random.randint(0, 29)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                valid = True
                for i, p in enumerate(new_points):
                    if i == idx:
                        continue
                    if p == (nx, ny):
                        valid = False
                        break
                if valid:
                    new_points[idx] = (nx, ny)
                else:
                    continue
