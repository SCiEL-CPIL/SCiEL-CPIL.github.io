            # Try swapping two random columns or rows
            if random.random() < 0.5:
                c1, c2 = random.sample(range(n), 2)
                new_L = [row[:] for row in best_L]
                for i in range(n):
                    new_L[i][c1], new_L[i][c2] = new_L[i][c2], new_L[i][c1]
            else:
                r1, r2 = random.sample(range(n), 2)
                new_L = [row[:] for row in best_L]
                new_L[r1], new_L[r2] = new_L[r2], new_L[r1]
