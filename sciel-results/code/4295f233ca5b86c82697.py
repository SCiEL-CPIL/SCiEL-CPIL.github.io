# Precompute mod table
MOD_TABLE = [[0] * (2 * N + 1) for _ in range(N + 1)]
for a in range(1, N + 1):
    for x in range(2 * N + 1):
        MOD_TABLE[a][x] = x % a
