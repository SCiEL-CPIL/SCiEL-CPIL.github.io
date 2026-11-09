# Correct Grötzsch construction
add_edge(0, 1)
add_edge(1, 2)
add_edge(2, 3)
add_edge(3, 4)
add_edge(4, 0)

# Primes
# 5 (0') -> 1, 4
add_edge(5, 1)
add_edge(5, 4)

# 6 (1') -> 0, 2
add_edge(6, 0)
add_edge(6, 2)

# 7 (2') -> 1, 3
add_edge(7, 1)
add_edge(7, 3)

# 8 (3') -> 2, 4
add_edge(8, 2)
add_edge(8, 4)

# 9 (4') -> 3, 0
add_edge(9, 3)
add_edge(9, 0)

# Center
add_edge(10, 5)
add_edge(10, 6)
add_edge(10, 7)
add_edge(10, 8)
add_edge(10, 9)
