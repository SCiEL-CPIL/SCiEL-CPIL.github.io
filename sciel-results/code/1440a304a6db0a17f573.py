import array

# Constants
NUM_ITEMS = 400
BYTES_PER_ITEM = 2
COUNTER_BYTES = NUM_ITEMS * BYTES_PER_ITEM  # 800 bytes
STATE_SIZE = 1024
PAD_BYTES = STATE_SIZE - COUNTER_BYTES  # 224 bytes

def update(state, item):
    """
    Updates the frequency table state with a new item.
    
    Args:
        state: bytes (1024 bytes)
        item: int (0..399)
        
    Returns:
        bytes: New state (1024 bytes)
    """
    # Unpack the first 800 bytes into an array of 400 unsigned shorts
    counters = array.array('H', state[:COUNTER_BYTES])
    
    # Increment the count for the given item
    counters[item] += 1
    
    # Pack the array back to bytes
    new_counters_bytes = counters.tobytes()
    
    # Pad with zeros to reach 1024 bytes
    return new_counters_bytes + b'\x00' * PAD_BYTES

def query(state, item):
    """
    Queries the frequency estimate for an item.
    
    Args:
        state: bytes (1024 bytes)
        item: int (0..399)
        
    Returns:
        int: Frequency estimate
    """
    # Unpack the first 800 bytes
    counters = array.array('H', state[:COUNTER_BYTES])
    
    # Return the count for the given item
    return counters[item]
