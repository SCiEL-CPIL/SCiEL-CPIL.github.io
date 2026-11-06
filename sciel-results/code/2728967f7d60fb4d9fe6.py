def design(problem):
    size = problem["pixels"]
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Checkerboard pattern
            # 1 for dielectric, 0 for air
            if (i + j) % 2 == 0:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell
