    def get_b_indices(block, sub_block):
        if block == 'T':
            row_offset = 0
        else:
            row_offset = 2
            
        if sub_block < 3:
            col_offset = sub_block * 2
            return {0: row_offset * p + col_offset, 
                    1: row_offset * p + col_offset + 1,
                    2: (row_offset + 1) * p + col_offset,
                    3: (row_offset + 1) * p + col_offset + 1}
        else:
            col_offset = 6
            return {0: row_offset * p + col_offset,
                    1: (row_offset + 1) * p + col_offset}
