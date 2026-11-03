r=0: (A00+A11)(B00+B11) → C00, C11  [Strassen M1]
r=1: (A10+A11)B00 → C10, -C11  [Strassen M2]
r=2: A00(B01-B11) → C01, C11  [Strassen M3]
r=3: A11(B10-B00) → C00, C10  [Strassen M4]
r=4: (A00+A01)B11 → -C00, C01  [Strassen M5]
r=5: (A10-A00)(B00+B01) → C11  [Strassen M6]
r=6: (A01-A11)(B10+B11) → C00  [Strassen M7]
r=7: A02*B20 → C00
r=8: A02*B21 → C01
r=9: A12*B20 → C10
r=10: A12*B21 → C11
r=11: (A00+A10)*B02 → C02, C12
r=12: (A01+A11)*B12 → C02, C12
r=13: (A02+A12)*B22 → C02, C12
r=14: A00*B02 → C02 (wait, this overlaps with r=11)
