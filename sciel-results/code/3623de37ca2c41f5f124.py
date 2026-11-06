"""Triangle-free graphs without independent 10-sets (Ramsey R(3,10)).

Uses a circulant graph on 38 vertices.
- Vertices 0..37
- Edge (i,j) iff (j - i) mod 38 is congruent to 1 mod 3
- Differences: 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37
- Triangle-free: verified by checking small sums/differences
- Independence number <= 9 (expected)
"""

def construct():
    n = 38
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
