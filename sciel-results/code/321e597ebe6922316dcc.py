rank = 28
add_strassen(0, 1, 0, 1, 0, 1)  # 7 terms
add_strassen(0, 1, 0, 1, 2, 3)  # 7 terms
add_schoolbook([0, 1], [0, 1], [4])  # 2*1*1 = 2 terms
add_schoolbook([0, 1], [2], [0, 1])  # 2*1*2 = 4 terms
add_schoolbook([0, 1], [2], [2, 3])  # 2*1*2 = 4 terms
add_schoolbook([0, 1], [2], [4])  # 2*1*1 = 2 terms
