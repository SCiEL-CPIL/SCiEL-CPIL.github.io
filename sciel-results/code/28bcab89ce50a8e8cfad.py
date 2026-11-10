def design(problem):
    size = problem["pixels"]
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Air cross: |x| < 0.15 or |y| < 0.15
            if abs(x) < 0.15 or abs(y) < 0.15:
                row.append(0)
            else:
                row.append(1)
        cell.append(row)
    return cell
