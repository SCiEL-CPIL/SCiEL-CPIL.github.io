    while current_n < 62:
        if time.time() > deadline:
            break

        extended = False
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            rng = random.Random()
            # Create a new matrix of size (current_n + 1)
            new_matrix = [row[:] + [-1] for row in current_matrix]
            new_matrix.append([-1] * (current_n + 1))
            
            if try_extend(new_matrix, current_n, rng):
                current_matrix = new_matrix
                extended = True
                break

        if extended:
            current_n += 1
            if current_n > best_n:
                best_n = current_n
                best_matrix = [row[:] for row in current_matrix]
        else:
            break
