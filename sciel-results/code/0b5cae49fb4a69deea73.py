    def get_block_indices(row_offset, col_offset, size=4):
        return [
            row_offset * size + col_offset,
            row_offset * size + col_offset + 1,
            (row_offset + 1) * size + col_offset,
            (row_offset + 1) * size + col_offset + 1
        ]
    