            for j in range(n):
                if j == i:
                    continue
                # x_i = x_j + (a_j - a_i - 1/n)*y + 1/n
                # Sample at several y values
                for y_sample in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    cp1 = best_x[j] + (a[j] - a[i] - 1.0/n) * y_sample + 1.0/n
                    cp2 = best_x[j] + (a[j] - a_i + 1.0/n) * y_sample - 1.0/n
