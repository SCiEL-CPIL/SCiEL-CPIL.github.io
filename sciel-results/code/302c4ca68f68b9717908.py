W = [[0] * rank for _ in range(cols * rows)]
# ...
for i in range(rows):
    for j in range(inner):
        for k in range(cols):
            # ...
            W[k * rows + i][col] = 1
