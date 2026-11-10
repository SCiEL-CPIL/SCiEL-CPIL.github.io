def design(problem):
        size = problem["pixels"]
        cell = []
        # Try a square rod of side 0.55
        # Half side = 0.275
        half_side = 0.275
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_side and abs(y) < half_side:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
